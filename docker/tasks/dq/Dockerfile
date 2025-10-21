FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir great-expectations==0.18.14 pandas==2.2.2
COPY docker/tasks/dq/validate.py /app/validate.py
ENTRYPOINT ["python","/app/validate.py"]
