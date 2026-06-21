# Use an official lightweight Python image
FROM python:3.9-slim

WORKDIR /code

# Install system dependencies required for OpenCV/YOLO
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new non-root user (required by Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the rest of your application code
COPY --chown=user . $HOME/app

# Expose the specific port Hugging Face expects (7860) and disable XSRF for smooth file uploads
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
