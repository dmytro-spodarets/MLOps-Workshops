from ultralytics import YOLO
from ultralytics import settings
import torch

# Перевірка наявності пристрою GPU та MPS (для Macbook)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Завантаження базової моделі YOLOv8 з конфігураційного файлу 'yolov8n.yaml'
model = YOLO('yolov8n.yaml')

# Оновлення налаштувань з вказівкою шляху до папки з датасетами
settings.update({"datasets_dir": "/Users/spodarets/GitHub/MLOps-Workshops/Deploy_and_Observability/data_drift/datasets"})

# Навчання моделі з використанням вказаного датасету
results = model.train(
    data='dataset.yaml',   # Шлях до конфігурації датасету
    epochs=150,            # Кількість епох навчання
    imgsz=224,             # Розмір зображень для навчання
    batch=16,              # Розмір батчу
    name='yolov8n_custom_v2', # Назва експерименту
    device=device.type,    # Пристрій для навчання (CPU або MPS)
    pretrained=False       # Чи використовувати попередньо навчену модель
)
