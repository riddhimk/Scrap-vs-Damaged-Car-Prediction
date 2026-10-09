# Use the official Python slim image for a smaller footprint
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV or other packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the rest of the application
COPY . .

# Expose port (7860 is the default for Hugging Face Spaces)
EXPOSE 7860

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]

