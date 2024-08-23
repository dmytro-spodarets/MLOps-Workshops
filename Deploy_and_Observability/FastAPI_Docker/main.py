from typing import List
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn as nn
import torch

# Инициализация FastAPI
app = FastAPI()

# Класс для линейной регрессии
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Загрузка модели
model = LinearRegressionModel()
model.load_state_dict(torch.load('model/linear_regression_model.pth'))
model.eval()

# Класс для определения структуры входных данных
class CarYears(BaseModel):
    years: List[float]

# Маршрут для предсказаний
@app.post("/invocations")
def predict(car_years: CarYears):
    years_int = [int(year) for year in car_years.years]
    input_data = pd.DataFrame({'year': years_int})

    # Преобразование данных в тензор
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

    # Предсказания модели
    predictions = model(input_tensor).detach().numpy().flatten()

    prediction_data = pd.DataFrame({
        'year': years_int,
        'price': predictions.tolist()
    })

    print("Input years:", years_int)
    print("Predictions:", predictions.tolist())

    return prediction_data.to_dict(orient='records')

# Маршрут для проверки состояния сервиса
@app.get("/ping")
def ping():
    return {"status": "ok"}

# Запуск сервиса, если модуль запускается напрямую
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
