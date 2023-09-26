# Use an official Python runtime as base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Install Git LFS
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install

# Copy the requirements file for pip
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /app/

# Run Gunicorn
CMD ["gunicorn", "-w 4", "-b 0.0.0.0:8000", "main:app"]
