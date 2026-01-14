# Serendipity AI Backend - Production Dockerfile
# =================================================
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r serendipity && useradd -r -g serendipity serendipity

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/serendipity/.local
ENV PATH=/home/serendipity/.local/bin:$PATH

# Copy application code
COPY --chown=serendipity:serendipity . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R serendipity:serendipity /app

# Switch to non-root user
USER serendipity

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    DEBUG=false \
    HOST=0.0.0.0 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
