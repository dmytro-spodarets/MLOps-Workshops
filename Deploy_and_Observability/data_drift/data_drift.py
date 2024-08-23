from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
from ultralytics import YOLO
import os

# Завантаження навченої моделі YOLO з файлу 'model-v2.pt'.
model = YOLO('model-v2.pt')

# Функція для витягування ознак із зображення за допомогою моделі YOLO.
def extract_features(image_path, model):
    # Виконання передбачення з порогом впевненості 0.90.
    results = model(image_path, conf=0.90)
    features = []

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for i in range(len(results[0].boxes)):
            class_id = int(results[0].boxes.cls[i])
            object_type = model.names.get(class_id, 'unknown')
            features.append(object_type)
    else:
        features.append('unknown')

    return features

# Функція для створення набору даних ознак із папки зображень.
def create_feature_dataset(image_folder, model):
    features_list = []

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            features = extract_features(image_path, model)
            if features:
                for feature in features:
                    features_list.append({'object_type': feature})

    return pd.DataFrame(features_list)

# Шлях до референсного та поточного набору зображень.
reference_dataset = 'datasets/synthetic_dataset_v2/images'
current_dataset = 'datasets/current_dataset_v2'

# Створення наборів ознак для референсного та поточного набору.
reference_features = create_feature_dataset(reference_dataset, model)
current_features = create_feature_dataset(current_dataset, model)

# Перевірка, чи датафрейми не порожні.
if not reference_features.empty and not current_features.empty:
    # Налаштування звіту про дріфт даних.
    column_mapping = ColumnMapping()
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features, column_mapping=column_mapping)

    # Збереження звіту у форматі HTML та JSON.
    report.save_html('drift_report.html')
    report.save("snapshot.json")
else:
    print("Недостатньо даних для створення звіту про дріфт.")
