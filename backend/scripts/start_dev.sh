#!/bin/bash
# Development startup script

echo "🚀 Starting Flowza Development Environment"

# Start Docker services
echo "📦 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
docker-compose ps

# Activate Python environment and start FastAPI
echo "🐍 Starting FastAPI server..."
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &

echo "✅ Development environment ready!"
echo "📊 API: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
echo "🗄️ PostgreSQL: localhost:5433"
echo "🔄 Redis: localhost:6380"