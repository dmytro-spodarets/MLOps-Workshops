import os
import json
import random
from typing import Dict, List
import time
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# Ігнорування попереджень
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ініціалізація wandb з новим бекендом
wandb.require("core")
run = wandb.init(project="t5-qa-training", name="t5-small-qa")

# Налаштування пристроїв
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Використовується пристрій: {device}")
print(f"Кількість доступних GPU: {n_gpu}")

# Налаштування параметрів
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 4 * max(1, n_gpu)
VALID_BATCH_SIZE = 4 * max(1, n_gpu)
EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
MAX_GRAD_NORM = 1.0


# Завантаження та підготовка даних
def load_json_files(directory: str) -> List[Dict]:
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data.extend(json.load(f))
    return data


print("Завантаження даних...")
all_data = load_json_files('devops-qa-dataset')
random.shuffle(all_data)

# Розділення даних
train_data, temp_data = train_test_split(all_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Розміри датасетів: Навчальний - {len(train_data)}, Валідаційний - {len(val_data)}, Тестовий - {len(test_data)}")


# Збереження наборів даних
def save_dataset(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


save_dataset(train_data, 'train_dataset.json')
save_dataset(val_data, 'val_dataset.json')
save_dataset(test_data, 'test_dataset.json')

print("Набори даних збережено.")

# Логування інформації про датасет в W&B
wandb.log({
    "train_size": len(train_data),
    "val_size": len(val_data),
    "test_size": len(test_data)
})

# Створення артефакту датасету в W&B
dataset_artifact = wandb.Artifact("devops-qa-dataset", type="dataset")
dataset_artifact.add_file("train_dataset.json")
dataset_artifact.add_file("val_dataset.json")
dataset_artifact.add_file("test_dataset.json")
run.log_artifact(dataset_artifact)


# Створення класу датасету
class QADataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: T5Tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        question = item['question']
        answer = item['answer']

        source = self.tokenizer.encode_plus(f"question: {question}",
                                            max_length=MAX_LENGTH,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt')

        target = self.tokenizer.encode_plus(answer,
                                            max_length=MAX_LENGTH,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt')

        return {
            'source_ids': source['input_ids'].squeeze(),
            'source_mask': source['attention_mask'].squeeze(),
            'target_ids': target['input_ids'].squeeze(),
            'target_mask': target['attention_mask'].squeeze()
        }


# Ініціалізація моделі та токенізатора
print("Ініціалізація моделі та токенізатора...")
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('t5-small')
print("Модель та токенізатор успішно завантажено.")

# Переміщення моделі на GPU та налаштування для multi-GPU
model = model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Створення датасетів та даталоадерів
print("Створення датасетів та даталоадерів...")
train_dataset = QADataset(train_data, tokenizer)
val_dataset = QADataset(val_data, tokenizer)
test_dataset = QADataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

# Оптимізатор та планувальник
print("Налаштування оптимізатора та планувальника...")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=WARMUP_STEPS,
                                            num_training_steps=len(train_loader) * EPOCHS)


# Функція навчання
def train(epoch):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Епоха {epoch + 1}/{EPOCHS} [Навчання]")
    for batch_idx, batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['source_ids'].to(device)
        attention_mask = batch['source_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_mask = batch['target_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids,
                        decoder_attention_mask=target_mask)
        loss = outputs.loss

        if n_gpu > 1:
            loss = loss.mean()

        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        wandb.log({"train_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# Функція валідації
def validate(loader, phase="val"):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"{phase.capitalize()}")
        for batch in progress_bar:
            input_ids = batch['source_ids'].to(device)
            attention_mask = batch['source_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids,
                            decoder_attention_mask=target_mask)
            loss = outputs.loss

            if n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


# Основний цикл навчання
print("Початок навчання...")
best_val_loss = float('inf')
best_model = None

for epoch in range(EPOCHS):
    print(f"\nПочаток Епохи {epoch + 1}/{EPOCHS}")
    train_loss = train(epoch)
    val_loss = validate(val_loader, "val")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(f"Епоха {epoch + 1} завершена. Навчальні втрати: {train_loss:.4f}, Валідаційні втрати: {val_loss:.4f}")

    # Збереження найкращої моделі
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.module if n_gpu > 1 else model
        print(f"Нова найкраща модель з валідаційними втратами: {val_loss:.4f}")

# Фінальне тестування
print("\nФінальне тестування...")
test_loss = validate(test_loader, "test")
wandb.log({"test_loss": test_loss})

# Збереження всіх необхідних даних
print("Збереження моделі та додаткових даних...")
save_directory = "t5_qa_model_final"
os.makedirs(save_directory, exist_ok=True)

# Збереження моделі
best_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Збереження конфігурації та метаданих
config = {
    "model_name": "t5-small-qa",
    "max_length": MAX_LENGTH,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "valid_batch_size": VALID_BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "best_val_loss": best_val_loss,
    "test_loss": test_loss,
    "device": str(device),
    "n_gpu": n_gpu
}

with open(os.path.join(save_directory, "config.json"), "w") as f:
    json.dump(config, f)

# Створення та збереження артефакту W&B
artifact = wandb.Artifact(f"t5-qa-model-final", type="model")
artifact.add_dir(save_directory)
run.log_artifact(artifact)

print(f"Модель та всі необхідні дані збережено у {save_directory} та W&B")

wandb.finish()
print("Навчання завершено!")