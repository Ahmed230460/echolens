import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, Blueprint, flash, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import google.generativeai as genai
from PIL import Image
import io
import time
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # For flash messages

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Blueprint for main routes
main_bp = Blueprint('main', __name__)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes for static pages
@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/features')
def features():
    return render_template('features.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        try:
            # Send email (configure your SMTP server)
            msg = MIMEText(f"Name: {name}\nEmail: {email}\nMessage: {message}")
            msg['Subject'] = 'Contact Form Submission - EchoLens'
            msg['From'] = os.getenv("SMTP_EMAIL")
            msg['To'] = os.getenv("SMTP_EMAIL")
            with smtplib.SMTP(os.getenv("SMTP_SERVER"), os.getenv("SMTP_PORT")) as server:
                server.starttls()
                server.login(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
                server.send_message(msg)
            flash('Thank you for your message! We will get back to you soon.', 'success')
            return redirect(url_for('main.contact'))
        except Exception as e:
            flash(f'Error sending message: {str(e)}', 'error')
            return redirect(url_for('main.contact'))
    return render_template('contact.html')

@main_bp.route('/analysis')
def analysis():
    return render_template('analysis.html')

# Preprocess video
def preprocess_video(input_path, output_path, target_size=(224, 224)):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        normalized_frame = resized_frame / 255.0
        output_frame = (normalized_frame * 255).astype(np.uint8)
        out.write(output_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Keyframe extraction
def extract_keyframes(video_path, output_video_raw, output_video_annotated, output_video_significant):
    model = YOLO("yolo12n.pt")
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0, []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_raw = cv2.VideoWriter(output_video_raw, fourcc, fps, (frame_width, frame_height))
    out_annotated = cv2.VideoWriter(output_video_annotated, fourcc, fps, (frame_width, frame_height))
    out_significant = cv2.VideoWriter(output_video_significant, fourcc, fps, (frame_width, frame_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    event_active = False
    event_start_frame = None
    no_motion_threshold = 30
    motion_history = []
    saved_frames = 0
    saved_significant_frames = 0
    events = []
    event_motion_peak = 0
    peak_frame = None
    frames_per_keyframe = 10
    significant_frames_per_keyframe = 1

    def frame_to_time(frame_num, fps):
        return frame_num / fps

    def check_tracking_event(results):
        if not results or not hasattr(results[0], 'boxes'):
            return False
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x3, y3, x4, y4 = boxes[j]
                if (x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3):
                    return True
        return len(boxes) > 0

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            if event_active:
                end_time = frame_to_time(frame_count, fps)
                events[-1]["end_frame"] = frame_count
                events[-1]["end_time"] = end_time
                if peak_frame is not None:
                    for _ in range(frames_per_keyframe):
                        out_raw.write(peak_frame[0])
                        out_annotated.write(peak_frame[1])
                    saved_frames += 1
                    for _ in range(significant_frames_per_keyframe):
                        out_significant.write(peak_frame[0])
                    saved_significant_frames += 1
                event_active = False
            break
        frame_count += 1
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(prev_gray, current_gray)
        _, thresh_gray = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        non_zero_count = cv2.countNonZero(thresh_gray)
        motion_history.append(non_zero_count)
        if len(motion_history) > 50:
            motion_history.pop(0)
        adaptive_threshold = max(100, np.mean(motion_history) * 2)
        diff_b = cv2.absdiff(prev_frame[:, :, 0], current_frame[:, :, 0])
        diff_g = cv2.absdiff(prev_frame[:, :, 1], current_frame[:, :, 1])
        diff_r = cv2.absdiff(prev_frame[:, :, 2], current_frame[:, :, 2])
        color_diff = cv2.max(cv2.max(diff_b, diff_g), diff_r)
        _, thresh_color = cv2.threshold(color_diff, 30, 255, cv2.THRESH_BINARY)
        color_change = cv2.countNonZero(thresh_color)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_motion = np.mean(magnitude) > 2
        motion_detected = (non_zero_count > adaptive_threshold) or \
                          (color_change > adaptive_threshold) or \
                          flow_motion
        tracking_event = False
        annotated_frame = current_frame.copy()
        if frame_count % 5 == 0:
            results = model.track(
                source=current_frame,
                persist=True,
                tracker="botsort.yaml",
                conf=0.3,
                iou=0.5,
                verbose=False
            )
            tracking_event = check_tracking_event(results)
            if results and results[0].boxes:
                annotated_frame = results[0].plot()
        significant_event = motion_detected or tracking_event
        if significant_event:
            if not event_active:
                event_active = True
                event_start_frame = frame_count
                start_time = frame_to_time(frame_count, fps)
                events.append({"start_frame": frame_count, "start_time": start_time})
                event_motion_peak = 0
                peak_frame = None
            motion_score = non_zero_count + color_change + np.mean(magnitude)
            if motion_score > event_motion_peak:
                event_motion_peak = motion_score
                peak_frame = (current_frame, annotated_frame)
        elif event_active and (frame_count - event_start_frame) > no_motion_threshold:
            event_active = False
            end_time = frame_to_time(frame_count, fps)
            events[-1]["end_frame"] = frame_count
            events[-1]["end_time"] = end_time
            if peak_frame is not None:
                for _ in range(frames_per_keyframe):
                    out_raw.write(peak_frame[0])
                    out_annotated.write(peak_frame[1])
                saved_frames += 1
                for _ in range(significant_frames_per_keyframe):
                    out_significant.write(peak_frame[0])
                saved_significant_frames += 1
        prev_gray = current_gray
        prev_frame = current_frame.copy()
    cap.release()
    out_raw.release()
    out_annotated.release()
    out_significant.release()
    return saved_significant_frames, events

# I3D Classifier
def load_i3d_ucf_finetuned(repo_id="Ahmeddawood0001/i3d_ucf_finetuned", filename="i3d_ucf_finetuned.pth"):
    class I3DClassifier(nn.Module):
        def __init__(self, num_classes):
            super(I3DClassifier, self).__init__()
            self.i3d = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
            self.dropout = nn.Dropout(0.3)
            self.i3d.blocks[6].proj = nn.Linear(2048, num_classes)
        def forward(self, x):
            x = self.i3d(x)
            x = self.dropout(x)
            return x
    device = torch.device("cpu")
    model = I3DClassifier(num_classes=8).to(device)
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def extract_frames(video_path, max_frames=32, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    while len(frames) < max_frames:
        frames.append(frames[-1])
    frames = frames[:max_frames]
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = frames.permute(1, 0, 2, 3)
    cap.release()
    return frames

def classify_video(video_path, model, labels):
    frames = extract_frames(video_path)
    frames = frames.unsqueeze(0).to(torch.device("cpu"))
    with torch.no_grad():
        outputs = model(frames)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = labels[predicted_idx]
        confidence = probabilities[0, predicted_idx].item()
    return predicted_label, confidence

# Frame extraction for Gemini
def extract_frames_for_gemini(video_path, output_dir, frame_rate=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frame_rate)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        frame_idx += 1
    cap.release()
    return frames

# Generate descriptions and summary with retry logic
def generate_descriptions_and_summary(frames, video_prediction):
    descriptions = {}
    max_retries = 3
    for frame_path in frames:
        prompt = (
            f"This frame is from a video classified as '{video_prediction}'. "
            "Describe the event happening in the image in one sentence."
        )
        with open(frame_path, "rb") as img_file:
            image_data = Image.open(io.BytesIO(img_file.read()))
        for attempt in range(max_retries):
            try:
                response = gemini_model.generate_content([prompt, image_data])
                descriptions[frame_path] = response.text
                break
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    retry_delay = 30
                    if hasattr(e, 'retry_delay') and hasattr(e.retry_delay, 'seconds'):
                        retry_delay = e.retry_delay.seconds
                    print(f"Quota exceeded, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    raise Exception("Gemini API quota exhausted after max retries.")
    summary_prompt = (
        "Here are multiple descriptions of frames from a surveillance video:\n"
        + "\n".join(descriptions.values()) +
        "\nBased on these descriptions, provide a concise summary of the overall event."
    )
    for attempt in range(max_retries):
        try:
            summary_response = gemini_model.generate_content(summary_prompt)
            return descriptions, summary_response.text
        except ResourceExhausted as e:
            if attempt < max_retries - 1:
                retry_delay = 30
                if hasattr(e, 'retry_delay') and hasattr(e.retry_delay, 'seconds'):
                    retry_delay = e.retry_delay.seconds
                print(f"Quota exceeded for summary, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise Exception("Gemini API quota exhausted for summary after max retries.")
    return descriptions, "Summary generation failed due to quota limits."

# Video analysis route
@main_bp.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        try:
            # Define output paths
            output_video_preprocessed = os.path.join(OUTPUT_FOLDER, "output_video_preprocessing.mp4")
            output_video_raw = os.path.join(OUTPUT_FOLDER, "keyframes_only_output.mp4")
            output_video_annotated = os.path.join(OUTPUT_FOLDER, "keyframes_annotated_output.mp4")
            output_video_significant = os.path.join(OUTPUT_FOLDER, "significant_keyframes_output.mp4")
            frames_dir = os.path.join(OUTPUT_FOLDER, "frames")

            # Step 1: Preprocess video
            preprocess_video(video_path, output_video_preprocessed)

            # Step 2: Extract keyframes
            saved_frames, events = extract_keyframes(
                output_video_preprocessed,
                output_video_raw,
                output_video_annotated,
                output_video_significant
            )

            # Step 3: Classify video
            labels = ["arrest", "Explosion", "Fight", "normal", "roadaccidents", "shooting", "Stealing", "vandalism"]
            i3d_model = load_i3d_ucf_finetuned()
            predicted_label, confidence = classify_video(output_video_significant, i3d_model, labels)

            # Step 4: Extract frames for Gemini
            frames = extract_frames_for_gemini(output_video_significant, frames_dir)

            # Step 5: Generate descriptions and summary
            descriptions, summary = generate_descriptions_and_summary(frames, predicted_label)

            # Prepare response (exclude descriptions)
            response = {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'summary': summary,
                'events': events
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            for frame in frames:
                if os.path.exists(frame):
                    os.remove(frame)

    return jsonify({'error': 'Invalid file type'}), 400

# Register blueprint
app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(debug=True)