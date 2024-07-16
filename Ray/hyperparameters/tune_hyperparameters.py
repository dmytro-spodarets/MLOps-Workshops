import ray
from ray import tune, air
from train_model import train_model
import os

# Ініціалізація Ray
ray.init(address='auto')

# Налаштування гіперпараметрів з використанням Ray Tune
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128])
}

# Вказуємо абсолютний шлях до директорії сховища
storage_path = os.path.abspath("./ray_results")

tuner = tune.Tuner(
    train_model,
    param_space=config,
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=10,
    ),
    run_config=air.RunConfig(
        name="tune_hyperparameters",
        storage_path=storage_path,
    ),
)

tuner.fit()
