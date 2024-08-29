import wandb
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Ініціалізація W&B
wandb.require("core")
run = wandb.init(project="t5-qa-training", job_type="model-download")

# Завантаження артефакту
artifact = run.use_artifact('dmytro-spodarets/model-registry/T5-DevOps-Chat:v1', type='model')
artifact_dir = artifact.download()

print(f"Артефакт завантажено до директорії: {artifact_dir}")

# Завантаження моделі та токенізатора
model = T5ForConditionalGeneration.from_pretrained(artifact_dir)
tokenizer = T5Tokenizer.from_pretrained(artifact_dir)

print("Модель та токенізатор успішно завантажено")

# Приклад використання моделі
input_text = "question: What is DevOps?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Вхідний текст: {input_text}")
print(f"Згенерована відповідь: {generated_text}")

# Збереження моделі та токенізатора локально
local_save_dir = "local_t5_model"
model.save_pretrained(local_save_dir)
tokenizer.save_pretrained(local_save_dir)

print(f"Модель та токенізатор збережено локально в директорії: {local_save_dir}")

wandb.finish()