FROM python:3.9.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV MONGODB_URI mongodb://mongo:BuQRyu9ToLd63jhYJhrx@103.150.92.14:27017/?authMechanism=DEFAULT

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
