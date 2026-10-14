# ---------- Base image ----------
# Pinned to a specific version; never use :latest
FROM python:3.12-slim

# ---------- Environment ----------
ENV PORT=8888 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------- System dependencies ----------
# curl: used by the HEALTHCHECK instruction
# wget: used by the notebook's dataset download utility
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl wget \
    && rm -rf /var/lib/apt/lists/*

# ---------- Non-root user ----------
# Create appuser and the /app working directory owned by it
RUN useradd -m -s /bin/bash appuser \
    && mkdir -p /app \
    && chown -R appuser:appuser /app

WORKDIR /app

# ---------- Python dependencies ----------
# Copy requirements first to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir jupyter

# ---------- Application code ----------
COPY --chown=appuser:appuser . .

# ---------- Runtime ----------
USER appuser

EXPOSE 8888

# Healthcheck uses curl (installed above). Jupyter's /api endpoint
# returns 200 once the notebook server is ready to accept traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# Run Jupyter Notebook.
# --allow-root is included per the project spec; it is a no-op when
# running as the non-root appuser, so it does not weaken security.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
