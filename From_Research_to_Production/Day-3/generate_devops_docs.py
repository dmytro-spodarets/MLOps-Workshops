from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Задайте API-ключ OpenAI

# Директория для сохранения документов
OUTPUT_DIR = "./devops-docs"

# Проверка существования директории, если нет — создаем
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Темы для документов
document_topics = [
    "Continuous Integration (CI) in DevOps",
    "Infrastructure as Code (IaC) with Terraform",
    "Monitoring and Logging in DevOps",
    "Microservices Architecture and DevOps",
    "Security and Compliance in DevOps"
]

# Функція для генерації тексту за допомогою OpenAI
def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert in DevOps and technical writing."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=2000,  # Орієнтовно для отримання потрібного об'єму тексту
    temperature=0.7)
    return response.choices[0].message.content

# Генерація документів
for i, topic in enumerate(document_topics):
    # Промпт для генерації тексту
    prompt = (f"Write a detailed article about '{topic}'. The article should be divided into 10 paragraphs, "
              "each paragraph should have 10 sentences. The language should be formal and informative.")

    # Генерація тексту
    print(f"Generating text for document {i + 1}: {topic}...")
    generated_text = generate_text(prompt)

    # Збереження згенерованого тексту до файлу
    output_file = os.path.join(OUTPUT_DIR, f"devops_document_{i + 1}.txt")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_text)

    print(f"Document saved to {output_file}")

print("All documents generated and saved successfully!")
