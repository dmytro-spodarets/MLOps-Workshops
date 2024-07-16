import ray
import torch
from train_model import train_model

# Ініціалізація Ray
ray.init(address='auto')

# Розподілене тренування з використанням Ray
@ray.remote
def remote_train_model(num_epochs, learning_rate):
    return train_model(num_epochs, learning_rate)

# Запуск тренування на декількох вузлах
num_epochs = 5
learning_rate = 0.001
num_workers = 2

futures = [remote_train_model.remote(num_epochs, learning_rate) for _ in range(num_workers)]
results = ray.get(futures)

# Агрегація результатів
final_model_state = results[0]  # В даному прикладі ми просто беремо результат з одного з вузлів

# Збереження моделі
torch.save(final_model_state, "trained_model.pth")

print("Training completed and model saved.")
