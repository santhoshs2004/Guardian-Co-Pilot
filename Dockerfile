FROM python:3.10.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application code (including src folder)
COPY . .

# Expose port
EXPOSE 8000

# Start command - pointing to src/api.py
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]