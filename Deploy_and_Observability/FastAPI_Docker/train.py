import wandb
import torch
import torch.nn as nn
import torch.optim as optim

# Ініціалізація W&B
wandb.require("core")

# Створення нового W&B проекту для відстеження тренувань моделі
run = wandb.init(project='car_price_prediction', name='linear_regression')

# Дані про роки та ціни на автомобілі
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
prices = [10000, 9500, 9000, 8500, 8000, 7500, 7000, 6500, 6000, 5500, 5000]

# Перетворення даних у тензори для використання у моделі
X = torch.tensor(years, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(prices, dtype=torch.float32).view(-1, 1)

# Оголошення класу лінійної регресійної моделі
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Додавання лінійного шару
        self.linear = nn.Linear(1, 1)

    # Визначення прямого проходу даних через модель
    def forward(self, x):
        return self.linear(x)

# Ініціалізація моделі, функції втрат та оптимізатора
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0000001)

# Відстеження параметрів моделі у W&B
wandb.watch(model, log_freq=100)

# Навчання моделі протягом 10000 епох
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    # Логування втрат у W&B
    wandb.log({'epoch': epoch, 'loss': loss.item()})

# Збереження параметрів моделі у файл
torch.save(model.state_dict(), 'linear_regression_model.pth')

# Завантаження моделі як артефакту у W&B
artifact = wandb.Artifact('model', type='model')
artifact.add_file('linear_regression_model.pth')
run.log_artifact(artifact)
run.finish()
