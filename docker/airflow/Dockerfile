FROM apache/airflow:2.9.3-python3.11
USER root

# Install build tools and Docker client
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Add airflow user to docker group (GID 999 is common for docker)
RUN groupadd -g 999 docker || true && usermod -aG docker airflow

USER airflow
# Copy Airflow-specific requirements (not root requirements.txt)
COPY docker/airflow/requirements.txt /airflow-requirements.txt
RUN pip install --no-cache-dir -r /airflow-requirements.txt
