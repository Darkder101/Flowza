#!/bin/bash
# Start Celery worker

echo "🔄 Starting Celery worker..."

cd backend
source venv/bin/activate
celery -A app.services.task_executor.celery worker --loglevel=info