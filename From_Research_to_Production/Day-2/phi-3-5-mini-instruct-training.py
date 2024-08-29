# Імпорт необхідних бібліотек
import os
import json
import random
import time
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, get_linear_schedule_with_warmup, BitsAndBytesConfig
import wandb
import torch
from torch.utils.data import DataLoader
import psutil
import gc
from tqdm import tqdm

# Спроба імпортувати Unsloth для оптимізації
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    use_unsloth = True
    print("Використовуємо Unsloth для оптимізації")
except ImportError:
    from transformers import AutoModelForCausalLM
    use_unsloth = False
    print("Unsloth недоступний, використовуємо стандартний підхід")

# Використання нового бекенду wandb
wandb.require("core")

# Вимкнення паралелізму токенізатора
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Функція для виведення інформації про використання пам'яті
def print_memory_usage():
    print(f"Пам'ять CPU: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            max_allocated = torch.cuda.max_memory_allocated(i)
            if max_allocated > 0:
                print(f"Пам'ять GPU {i}: {allocated / max_allocated * 100:.2f}%")
            else:
                print(f"Пам'ять GPU {i}: 0.00%")

# Перевірка доступності CUDA та кількості GPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA доступна. Використовуємо {num_gpus} GPU.")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA недоступна. Використовуємо CPU.")

# Ініціалізація wandb для логування
wandb.init(project="phi-3.5-mini-instruct-training-unsloth")

# Шлях до директорії з файлами JSON
dataset_dir = "devops-qa-dataset"

# Об'єднання всіх JSON файлів в один список та фільтрація полів
combined_data = []

# Функція для фільтрації даних: залишаємо тільки "question" та "answer"
def filter_data(data):
    filtered_data = []
    for item in data:
        filtered_item = {
            "question": item.get("question", "")[:128],  # Обмежуємо довжину питання
            "answer": item.get("answer", "")[:128]  # Обмежуємо довжину відповіді
        }
        filtered_data.append(filtered_item)
    return filtered_data

# Читання всіх JSON файлів та об'єднання даних
for filename in os.listdir(dataset_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            filtered_data = filter_data(data)
            combined_data.extend(filtered_data)

print(f"Загальна кількість прикладів: {len(combined_data)}")

# Перемішування даних для випадкового розділення
random.shuffle(combined_data)

# Обмеження кількості прикладів для тестування
max_examples = 5000
combined_data = combined_data[:max_examples]

# Розділення на тренувальний, валідаційний та тестовий набори
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

print(f"Розмір тренувального набору: {len(train_data)}")
print(f"Розмір валідаційного набору: {len(val_data)}")
print(f"Розмір тестового набору: {len(test_data)}")

# Збереження розділених даних в wandb
wandb.log({"train_data_size": len(train_data), "val_data_size": len(val_data), "test_data_size": len(test_data)})

# Налаштування квантизації
quantization_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Завантаження моделі та токенізатора
model_name = "microsoft/Phi-3.5-mini-instruct"
if use_unsloth:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_sequence_length=256,
        dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    chat_template = get_chat_template("phi")
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Розмір моделі: {sum(p.numel() for p in model.parameters())} параметрів")

# Функція для підготовки даних
def prepare_data(examples):
    questions = examples["question"]
    answers = examples["answer"]
    if use_unsloth:
        prompts = [
            chat_template.format_messages([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ])
            for q, a in zip(questions, answers)
        ]
    else:
        prompts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, answers)]
    return tokenizer(prompts, truncation=True, padding=True, max_length=256)

# Створення датасету
dataset = Dataset.from_dict({"question": [item["question"] for item in combined_data],
                             "answer": [item["answer"] for item in combined_data]})

# Застосування попередньої обробки до наборів даних
tokenized_dataset = dataset.map(prepare_data, batched=True, remove_columns=dataset.column_names)

# Розділення датасету на train, validation, test
train_dataset = tokenized_dataset.shuffle(seed=42).select(range(len(train_data)))
val_dataset = tokenized_dataset.shuffle(seed=42).select(range(len(train_data), len(train_data) + len(val_data)))
test_dataset = tokenized_dataset.shuffle(seed=42).select(range(len(train_data) + len(val_data), len(combined_data)))

# Налаштування аргументів навчання
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=7,
    per_device_eval_batch_size=5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    gradient_accumulation_steps=32,
    fp16=False,
    bf16=True,
    report_to=["wandb"],
    dataloader_num_workers=0,
    learning_rate=5e-6,
    max_grad_norm=0.5,
)

# Налаштування моделі для навчання
if use_unsloth:
    model = FastLanguageModel.prepare_for_training(model, use_gradient_checkpointing=True)
    optimizer = FastLanguageModel.get_optimizer(model, lr=5e-6, weight_decay=0.01)
else:
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

# Створення колатора даних
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Функція для обчислення втрат
def compute_loss(batch, model):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
    return outputs.loss

# Функція для оцінки моделі
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = compute_loss(batch, model)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Створення DataLoader'ів
train_dataloader = DataLoader(
    train_dataset,
    batch_size=training_args.per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=training_args.per_device_eval_batch_size,
    collate_fn=data_collator
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=training_args.per_device_eval_batch_size,
    collate_fn=data_collator
)

# Створення планувальника швидкості навчання
num_training_steps = len(train_dataloader) * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

# Функція для виявлення аномальних градієнтів
def detect_anomaly(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Виявлено NaN або Inf градієнт у {name}")
                return True
    return False

# Цикл навчання
print("Початок навчання...")
for epoch in range(int(training_args.num_train_epochs)):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Епоха {epoch + 1}/{training_args.num_train_epochs}")):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = compute_loss(batch, model)
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if detect_anomaly(model):
            print(f"Виявлено аномалію на епосі {epoch + 1}, кроці {step}. Пропускаємо цей батч.")
            optimizer.zero_grad()
            continue

        if (step + 1) % training_args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if step % training_args.logging_steps == 0:
            print(f"Епоха {epoch + 1}, Крок {step}, Втрата: {loss.item() * training_args.gradient_accumulation_steps:.4f}")
            wandb.log({"train_loss": loss.item() * training_args.gradient_accumulation_steps, "epoch": epoch + 1, "step": step})
            print_memory_usage()

    avg_train_loss = total_loss / len(train_dataloader) * training_args.gradient_accumulation_steps
    print(f"Середня втрата на тренуванні: {avg_train_loss:.4f}")

    val_loss = evaluate(model, val_dataloader)
    print(f"Втрата на валідації: {val_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_train_loss, "val_loss": val_loss})

    # Очищення кешу CUDA
    torch.cuda.empty_cache()
    gc.collect()

    # Збереження моделі
    model.save_pretrained(f"./saved_model_epoch_{epoch + 1}")
    tokenizer.save_pretrained(f"./saved_model_epoch_{epoch + 1}")

# Оцінка на тестовому наборі
test_loss = evaluate(model, test_dataloader)
print(f"Втрата на тесті: {test_loss:.4f}")
wandb.log({"test_loss": test_loss})

# Збереження фінальної моделі
model.save_pretrained("./final_saved_model")
tokenizer.save_pretrained("./final_saved_model")

# Створення артефакту wandb та завантаження моделі
artifact = wandb.Artifact("phi_3_5_model_unsloth", type="model")
artifact.add_dir("./final_saved_model")
wandb.log_artifact(artifact)
print("Модель успішно завантажена в wandb")

# Завершення сесії wandb
wandb.finish()