FROM python:3.11-slim

WORKDIR /project
ENV PYTHONPATH=/project

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .