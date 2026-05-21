FROM python:3.10-slim

# Системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y \
#    libgl1-mesa-glx \
#    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем зависимости (стандартным способом, качает CUDA версии)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файлы проекта
COPY original_repo/ /app/original_repo/
COPY validate.py /app/
COPY tests/ /app/tests/
COPY reference_depth.npy /app/
COPY checkpoints/ /app/checkpoints/

CMD ["python", "validate.py"]