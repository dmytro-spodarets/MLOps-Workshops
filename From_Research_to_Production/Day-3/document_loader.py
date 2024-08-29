import os
import json
from typing import List, Dict
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Змінна для зберігання історії оброблених файлів
PROCESSED_FILES_HISTORY = "processed_files_history.json"
# Директорія з документами для обробки
DOCUMENTS_DIRECTORY = "./devops-docs"

# Ініціалізація клієнта Qdrant і створення колекції для зберігання векторів
def init_qdrant():
    # Ініціалізація клієнта Qdrant з використанням API ключа з перемінної оточення
    client = QdrantClient(
        url="https://3fb2a758-19fe-4158-b88b-41f9259cdcca.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY")  # Перемістили API ключ в змінну оточення
    )
    # Ім'я колекції в Qdrant
    collection_name = "devops_docs"

    # Перевірка наявності колекції, якщо її немає — створюємо нову
    try:
        client.get_collection(collection_name)
    except Exception as e:
        print(f"Колекція не існує. Створюємо нову колекцію: {e}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Ініціалізація векторних ембеддінгів з OpenAI
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Використання API ключа OpenAI з перемінної оточення
    )
    # Ініціалізація сховища векторів Qdrant
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    return vectorstore

# Функція для обробки документа та поділу його на частини
def process_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    # Визначення розширення файлу
    _, file_extension = os.path.splitext(file_path)

    # Завантаження документа в залежності від його типу
    if file_extension.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Непідтримуваний тип файлу: {file_extension}")

    documents = loader.load()

    # Поділ тексту на частини з використанням RecursiveCharacterTextSplitter
    # Параметри: chunk_size - максимальний розмір кожного шматка, chunk_overlap - кількість символів, що перекриваються між шматками
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Використання функції len для визначення довжини тексту
        separators=["\n\n", "\n", " ", ""]  # Послідовність роздільників для поділу тексту на частини
    )
    # text_splitter.split_documents() розділяє текст на частини, намагаючись максимально використовувати великий роздільник.
    # Спочатку текст розбивається за роздільником "\n\n" (подвійний новий рядок). Якщо після цього частини більші за chunk_size,
    # то текст розбивається далі за роздільником "\n" (один новий рядок), і так далі до " " (пробіл) і "" (порожній рядок).
    # Це дозволяє отримати максимальні частини, не перевищуючи chunk_size, з мінімально можливими розрізаннями тексту.
    return text_splitter.split_documents(documents)

# Функція для завантаження історії оброблених файлів з файлу
def load_processed_files_history() -> Dict[str, str]:
    if os.path.exists(PROCESSED_FILES_HISTORY):
        with open(PROCESSED_FILES_HISTORY, 'r') as f:
            return json.load(f)
    return {}

# Функція для збереження історії оброблених файлів у файл
def save_processed_files_history(history: Dict[str, str]):
    with open(PROCESSED_FILES_HISTORY, 'w') as f:
        json.dump(history, f, indent=2)

# Функція для сканування директорії та обробки документів
def scan_and_process_documents(directory: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # Перевірка наявності директорії
    if not os.path.exists(directory):
        print(f"Помилка: Директорія {directory} не існує.")
        return

    # Ініціалізація сховища векторів
    vectorstore = init_qdrant()
    # Завантаження історії оброблених файлів
    processed_files = load_processed_files_history()

    # Проходження по всіх файлах у директорії
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_mod_time = str(os.path.getmtime(file_path))

            # Пропуск файлів, які вже були оброблені
            if file_path in processed_files and processed_files[file_path] == file_mod_time:
                print(f"Пропускаємо вже оброблений файл: {file_path}")
                continue

            # Обробка файлів типу .txt та .pdf
            try:
                _, file_extension = os.path.splitext(file)
                if file_extension.lower() in ['.txt', '.pdf']:
                    docs = process_document(file_path, chunk_size, chunk_overlap)
                    vectorstore.add_documents(docs)
                    processed_files[file_path] = file_mod_time
                    print(f"Успішно додано {len(docs)} частин з {file_path}")
                else:
                    print(f"Пропуск непідтримуваного типу файлу: {file_path}")
            except Exception as e:
                print(f"Помилка при обробці {file_path}: {str(e)}")

    # Збереження оновленої історії оброблених файлів
    save_processed_files_history(processed_files)

# Основна функція для запуску обробки документів
if __name__ == "__main__":
    scan_and_process_documents(DOCUMENTS_DIRECTORY)
