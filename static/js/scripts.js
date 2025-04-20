document.getElementById('uploadBtn').addEventListener('click', async () => {
    const videoInput = document.getElementById('video');
    if (!videoInput.files.length) {
        alert('Please select a video file.');
        return;
    }
    const file = videoInput.files[0];
    const formData = new FormData();
    formData.append('video', file);
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');

    // Simulate progress bar
    const progressBar = document.getElementById('progressBar');
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        progressBar.style.width = `${progress}%`;
        if (progress >= 90) clearInterval(interval);
    }, 1000);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        clearInterval(interval);
        progressBar.style.width = '100%';
        const result = await response.json();
        if (response.ok) {
            document.getElementById('predictedLabel').textContent = result.predicted_label;
            document.getElementById('confidence').textContent = result.confidence.toFixed(4);
            document.getElementById('summary').textContent = result.summary;
            const eventsList = document.getElementById('events');
            eventsList.innerHTML = '';
            result.events.forEach((event, index) => {
                const li = document.createElement('li');
                li.textContent = `Event ${index + 1}: Start ${event.start_time.toFixed(2)}s (Frame ${event.start_frame}), End ${event.end_time.toFixed(2)}s (Frame ${event.end_frame})`;
                eventsList.appendChild(li);
            });
            document.getElementById('results').classList.remove('hidden');
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('An error occurred: ' + error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
        progressBar.style.width = '0%';
    }
});