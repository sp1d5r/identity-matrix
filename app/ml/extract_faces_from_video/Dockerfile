# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Update the package lists for upgrades for packages and their dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

# Run app.py when the container launches
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 main:app
