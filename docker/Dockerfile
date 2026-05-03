# Базовый образ
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY app/ app/
COPY models/ models/

# Открываем порт
EXPOSE 5000

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Команда запуска
CMD ["python", "-m", "app.api"]