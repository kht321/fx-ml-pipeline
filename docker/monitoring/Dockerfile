FROM python:3.11-slim

WORKDIR /app

# copy requirements first so Docker layer caching works
COPY docker/monitoring/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# now copy the actual app code
COPY docker/monitoring/app.py /app/app.py
COPY docker/monitoring/generate.py /app/generate.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8050"]