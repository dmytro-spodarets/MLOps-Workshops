# Используем базовый образ Bitnami PyTorch
FROM bitnami/pytorch:2.3.1

# Переменные окружения для WANDB
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY

# Создаем директорию для кеша WANDB
RUN mkdir -p /app/.cache/wandb
ENV WANDB_CACHE_DIR=/app/.cache/wandb

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем необходимые файлы
COPY requirements.txt requirements.txt
COPY download_model.py download_model.py
COPY main.py main.py
COPY start.sh start.sh

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Загружаем модель из S3
RUN python download_model.py

# Дополнительно удаляем стандартный entrypoint Bitnami, если это необходимо
RUN rm /opt/bitnami/scripts/pytorch/entrypoint.sh

# Устанавливаем entrypoint на start.sh
ENTRYPOINT ["./start.sh"]
