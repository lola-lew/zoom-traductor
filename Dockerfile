FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget gnupg ca-certificates curl \
    pulseaudio pulseaudio-utils \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libasound2 libasound2-dev \
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 \
    libxcb1 libxkbcommon0 libx11-6 libxcomposite1 \
    libxdamage1 libxext6 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 \
    fonts-liberation xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install chromium
RUN playwright install-deps chromium

COPY . .

ENV HEADLESS=1
ENV PYTHONUNBUFFERED=1
ENV PULSE_RUNTIME_PATH=/run/pulse

CMD ["bash", "-c", "pulseaudio --start --exit-idle-time=-1 && python app.py"]
