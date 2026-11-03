FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first for fast builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies manifest and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Expose default Streamlit port
EXPOSE 8501

# Streamlit health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entry point for Streamlit application
ENTRYPOINT ["streamlit", "run", "src/termsheet_ai/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
