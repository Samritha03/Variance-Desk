# One container: builds the React app, then serves it and the API from FastAPI.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gve ./gve
COPY data ./data
COPY --from=web /web/dist ./web/dist
ENV PORT=8000
EXPOSE 8000
# Set ANTHROPIC_API_KEY as a secret in your host's dashboard. Never bake it into the image.
CMD ["sh", "-c", "uvicorn gve.api:app --host 0.0.0.0 --port ${PORT}"]
