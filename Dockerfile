FROM python:3.13-slim

LABEL org.opencontainers.image.title="ds001-pipeline"
LABEL org.opencontainers.image.description="DS-001 Passive Information Pipeline — collects, filters, and ingests articles"
LABEL org.opencontainers.image.source="https://github.com/sekai-dev-team/ds001-pipeline"
LABEL org.opencontainers.image.authors="sekai-dev-team"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pipeline/ pipeline/
COPY collect.py .

# Default command (overridden by Hermes cron with --mode flag)
ENTRYPOINT ["python", "/app/collect.py"]
CMD ["--mode", "daily"]
