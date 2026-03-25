FROM python:3.13-slim

# Create and switch to /app
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy the rest of the project into /app
COPY . .

# Expose port (documentation only)
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]