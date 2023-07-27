# Use python 3.9
FROM python:3.9.9-slim-buster

# Declare working direction
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt requirements.txt

# Install requirements.txt
RUN pip install -r requirements.txt

# Set environment variables
ENV MONGODB_URI mongodb://mongo:BuQRyu9ToLd63jhYJhrx@103.150.92.14:27017/?authMechanism=DEFAULT

# Copy all files in this projects to working directory
COPY . .

# Command to start server
CMD ["gunicorn", "-b", "0.0.0.0:7000", "app:app"]
