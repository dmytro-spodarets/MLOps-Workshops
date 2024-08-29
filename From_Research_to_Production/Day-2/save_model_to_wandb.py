import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import wandb

# Ініціалізація W&B
wandb.init(project="t5-qa-training", name="save-best-model")

# Шлях до локально збереженої моделі
local_model_path = "t5_qa_model_best.pt"

# Завантаження моделі та токенізатора
print("Завантаження моделі та токенізатора...")
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Завантаження ваг моделі
print(f"Завантаження ваг моделі з {local_model_path}")
model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))

# Збереження моделі та токенізатора
model_save_path = "t5_qa_model_wandb"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Створення артефакту W&B
artifact = wandb.Artifact('t5-qa-model', type='model')

# Додавання файлів моделі до артефакту
artifact.add_dir(model_save_path)

# Логування артефакту
wandb.log_artifact(artifact)

print("Модель успішно збережена в W&B")
wandb.finish()