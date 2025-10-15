FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas==2.2.2 scikit-learn==1.5.1
COPY train.py /app/train.py
ENTRYPOINT ["python","/app/train.py"]
