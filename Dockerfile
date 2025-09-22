FROM python:3.10-slim
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN mkdir -p hf_cache && chmod -R 777 hf_cache
CMD ["python", "app.py"]
