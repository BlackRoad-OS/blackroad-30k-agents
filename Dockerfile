FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.0 \
    sympy==1.12

# Copy agent code
COPY base_agent.py .

# Environment variables (overridable)
ENV AGENT_TYPE=quantum_mechanics
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Expose port
EXPOSE 8080

# Run agent
CMD ["python3", "base_agent.py"]
