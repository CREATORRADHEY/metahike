FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port for Hugging Face Spaces (default 7860 or 8000 depending on space, we'll use 8000 as defined in app.py or expose 8000)
# But HF Spaces Docker uses port 7860 by default. Let's make sure it runs on 7860.
ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
