FROM python:3.9-slim-buster

RUN pip install awscli
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]