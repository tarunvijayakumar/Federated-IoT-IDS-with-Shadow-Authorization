FROM python:3.11-slim
RUN pip install paho-mqtt==2.1.0 torch numpy --no-cache-dir
WORKDIR /app
COPY model.pt .
COPY inference.py .
CMD ["python", "inference.py"]
