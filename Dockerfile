# Используем легкий базовый образ с поддержкой Python
FROM python:3.10-slim

# Устанавливаем системные зависимости, необходимые для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем требования и устанавливаем Python-зависимости
# (убедитесь, что в requirements.txt есть torch, torchvision, opencv-python)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта, чекпоинты и тестовые данные
COPY . .

# Запускаем скрипт валидации при старте контейнера
CMD ["python", "validate.py"]