# Credit Approval ML Pipeline
# Multi-stage Docker build for production deployment

FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Development stage
# ============================================
FROM base AS development

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest pytest-cov

COPY . .

CMD ["python", "-m", "pytest", "tests/", "-v"]

# ============================================
# Production stage
# ============================================
FROM base AS production

# Create non-root user
RUN useradd --create-home appuser
USER appuser

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# Create output directories with correct permissions
RUN mkdir -p ml_pipeline_output/models \
    ml_pipeline_output/plots \
    ml_pipeline_output/results \
    ml_pipeline_output/logs \
    ml_pipeline_output/final_model \
    && chown -R appuser:appuser ml_pipeline_output

# Health check (simple check if the app exists)
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import src; print('System healthy')" || exit 1

# Default command
CMD ["python", "main.py"]

# ============================================
# Training stage
# ============================================
FROM production AS training

CMD ["python", "scripts/train.py"]

# ============================================
# Inference stage (lightweight)
# ============================================
FROM python:3.10-slim AS inference

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home appuser
USER appuser

# Install only inference dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

CMD ["python", "-c", "from src.serving import ModelPredictor; print('Inference ready')"]
