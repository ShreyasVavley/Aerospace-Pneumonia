# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app/api

# Create user 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and change ownership to user
COPY --chown=user:user api/ ./api/
COPY --chown=user:user ml/pneumonia_model.pth ./ml/

# Switch to the non-root user required by HF
USER user

# Expose the port (Hugging Face Spaces requires 7860)
EXPOSE 7860

# Command to run the application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
