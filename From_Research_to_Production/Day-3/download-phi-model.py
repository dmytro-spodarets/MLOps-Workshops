import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Завантаження моделі та токенізатора з Hugging Face
model_name = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Збереження моделі та токенізатора в локальну директорію
save_directory = "./model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Метадані моделі
meta_info = {
    "model_name": model_name,
    "description": "Це локально збережена версія моделі microsoft/Phi-3.5-mini-instruct."
}

# Збереження метаданих у JSON файл
import json
with open(f"{save_directory}/meta_info.json", "w") as f:
    json.dump(meta_info, f)