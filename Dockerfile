FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create streamlit config to skip prompts
RUN mkdir -p /root/.streamlit && \
    echo '[browser]\ngatherUsageStats = false\n[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"' > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
