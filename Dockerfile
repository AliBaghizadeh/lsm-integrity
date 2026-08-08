# Hugging Face Spaces (Docker SDK) target for app/demo_app.py.
# Docker Spaces must listen on port 7860 -- Streamlit's own SDK used 8501,
# but that SDK is deprecated on HF Spaces in favour of this Docker path.

FROM python:3.12-slim

WORKDIR /app

# libgomp1: LightGBM's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv \
    && uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH" \
    APP_MODE=demo

EXPOSE 7860

CMD ["streamlit", "run", "app/demo_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
