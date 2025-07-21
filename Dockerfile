# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# --- SUPER OBVIOUS DOCKERFILE MARKER ---
RUN echo "============== THIS IS A VERY OBVIOUS DOCKERFILE MARKER =============="
# --- END SUPER OBVIOUS DOCKERFILE MARKER ---

# --- DIAGNOSTIC STEP: Print requirements.txt content during build ---
RUN echo "--- Contents of requirements.txt during build ---" && cat requirements.txt && echo "---------------------------------------------------"
# --- END DIAGNOSTIC STEP ---

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose app port (Render uses $PORT, but this is good practice)
EXPOSE 8080

# Start the app (Render's Start Command will override this, but keep it standard)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
