import requests
import json
import uuid

# Базовий URL для доступу до API Evidently
base_url = "http://localhost:8000"

# Ім'я проєкту, який потрібно знайти або створити
project_name = "Shape detection and classification"

# Зчитування JSON-звіту з файлу
with open("snapshot.json", "r") as f:
    json_report = f.read()

# Функція для отримання або створення проєкту в Evidently
def get_or_create_project(base_url, project_name):
    # Запит на отримання списку існуючих проєктів
    response = requests.get(f"{base_url}/api/projects/")
    response.raise_for_status()
    projects = response.json()

    # Пошук проєкту з потрібною назвою
    for project in projects:
        if project["name"] == project_name:
            print(f"Проєкт знайдено: {project_name}")
            return project["id"]

    # Якщо проєкт не знайдено, створюється новий проєкт
    project_id = str(uuid.uuid4())
    project_data = {"id": project_id, "name": project_name}
    response = requests.post(
        f"{base_url}/api/projects/", data=json.dumps(project_data), headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    print(f"Проєкт успішно створений з ID: {project_id}")
    return project_id

# Функція для додавання знімку до проєкту
def add_snapshot_to_project(base_url, project_id, json_report):
    response = requests.post(
        f"{base_url}/api/projects/{project_id}/snapshots/",
        data=json_report,
        headers={"Content-Type": "application/json"}
    )

    try:
        response.raise_for_status()
        print("Знімок успішно доданий")
    except requests.exceptions.HTTPError as err:
        print(f"Помилка додавання знімку: {response.status_code} - {response.content}")
        raise err

# Отримання або створення проєкту і отримання його ID
project_id = get_or_create_project(base_url, project_name)

# Додавання знімку до проєкту
add_snapshot_to_project(base_url, project_id, json_report)
