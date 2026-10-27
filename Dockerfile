FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the 90-minute synthetic demo match (deterministic from the seed) and its analysis cache.
ARG DEMO_SEED=24
RUN python -m server.prepare --minutes 90 --seed ${DEMO_SEED}
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
