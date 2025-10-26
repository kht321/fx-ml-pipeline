FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas==2.2.2 scikit-learn==1.5.1
COPY docker/tasks/trainer/train.py /app/train.py
ENTRYPOINT ["python","/app/train.py"]
