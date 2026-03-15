FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install chromium
RUN playwright install-deps chromium

COPY . .

ENV HEADLESS=1
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
