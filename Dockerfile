# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Flask backend (serves API + built React app)
FROM python:3.9-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ .

# Copy React build output into dist/
COPY --from=frontend-builder /build/dist ./dist

EXPOSE 8085

CMD ["waitress-serve", "--host=0.0.0.0", "--port=8085", "main:app"]
