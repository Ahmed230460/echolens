# Use an official Python runtime as the base image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
