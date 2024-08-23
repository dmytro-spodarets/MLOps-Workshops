# Майстерня з MLOps: Розгортання та моніторинг ML моделей

## Необхідні пакети
```
pip install ultralytics evidently pandas wandb
brew install terraform - для MacOS
pip install "dstack[all]" -U
```
## Робота з Weights & Biases

https://docs.wandb.ai/ 

## Розгортання за допомогою FastAPI & Docker

Команди для білда контейнера та його запуску (локально):
```
export WANDB_API_KEY=ваш API KEY
```
Якщо контейнер не містить модель
```
docker build -t lr-model .

docker run -p 8080:8080 \
    --env WANDB_API_KEY=$(echo $WANDB_API_KEY) \
    --env WANDB_CACHE_DIR=/app/.cache \
    -v $(pwd)/model:/app/model \
    -v $(pwd)/.cache:/app/.cache \
    lr-model


http://localhost:8080/docs
```

Збираємо для випадку, якщо контейнер містить модель
```
docker build -t lr-model --build-arg WANDB_API_KEY=$(echo $WANDB_API_KEY) .
docker run -p 8080:8080 -it lr-model
```
Збираємо для необхідної архітектури (наприклад, якщо збираємо на M3, а деплоїти будемо в SageMaker)
```
docker buildx create --use
docker buildx inspect --bootstrap
docker buildx build --platform linux/amd64 -t lr-model --build-arg WANDB_API_KEY=$(echo $WANDB_API_KEY) --load .
```

Перевіряємо за допомогою наступної команди:
```
curl -X 'POST' \
  'http://localhost:8080/invocations/' \
  -H 'Content-Type: application/json' \
  -d '{"years": [2024,2002]}'
```

## Push контейнера до ECR
1. Створити репозиторій
2. Зібрати контейнер 
3. Додати тег `docker tag lr-model:latest 493395458839.dkr.ecr.us-east-1.amazonaws.com/linear-regression:latest`
4. Зробити push `docker push 493395458839.dkr.ecr.us-east-1.amazonaws.com/linear-regression:latest`

*493395458839 та us-east-1 - ви повинні вказати свій аккаунт та регіон.

## Розгортання за допомогою AppRunner

Збираємо контейнер з моделью, так як до AppRunner неможливо підключити сторедж
Робимо push до репозиторія
У файлі `ml-app.tf` прописуємо правильну адресу до імеджу

Виконуємо 
```
terraform init
terraform plan
terraform apply
```
Може появитись помилка створення ролі. Потрібно трошки почекати та повторити apply

Коли все буде задеплоєно ви побачите URL. Використовуючи цей URL ви зможете перевірити роботу моделі:
```
curl -X 'POST' \
  'http://URL/invocations/' \
  -H 'Content-Type: application/json' \
  -d '{"years": [2024,2002]}'
```
Щоб все видалити, виконайте команду:
```
terraform destroy
```

## Розгортання за допомогою SageMaker

Збираємо імедж під `linux/amd64`

У файлі `model.tf` вказуємо правильну адресу до імеджу

Деплоємо стандартними tf командами:
```
terraform init
terraform plan
terraform apply
```

Перевіряємо за допомогою наступної команди:
```
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name ml-app-endpoint \
    --body '{"years": [2010, 2015, 2020]}' \
    --content-type application/json \
    --region us-east-1 \
    --cli-binary-format raw-in-base64-out \
    output.json > /dev/null 2>&1
```

Щоб все видалити, виконайте команду:
```
terraform destroy
```

## Розгортання за допомогою dstack

Налаштування підключення до клауду - https://dstack.ai/docs/reference/server/config.yml/

Запуск dstack сервера - https://dstack.ai/docs/installation/
```
dstack server
```
Приклад - https://dstack.ai/docs/examples/llms/llama31/

Розгортання:
```
export HUGGING_FACE_HUB_TOKEN=ваш токен
dstack init
dstack apply -f task.dstack.yml
```
Коли все буде задеплоєно, ви зможете перевірити роботу моделі:
```
curl 127.0.0.1:8001/v1/chat/completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "What is Deep Learning?"
        }
      ],
      "max_tokens": 128
    }'
```
Щоб все видалити, виконайте команду:
```
dstack delete
```

## Детекція Data Drift
### Розгортання Evidently
```
docker-compose up -d
```
### Створення датасетів
```
python gen_current_dataset.py
python gen_synthetic_dataset.py
```
### Тренування декількох версій моделей
У файлі `dataset.yaml` потрібно вказати датасет, кількість класів та самі класи. 
Перша модель повинна бути натренована на 2-х класах, а друга на всіх.
```
python model_train.py
```
### Проведення єесперименту з детекції Data Drift
Для генерації репорту запускаємо 
```
python data_drift.py
```
Для відправлення репорту до Evidently виконуємо 
```
python data_drift_report.py
```
Щоб побачити Data Drift виконуємо запуск з наступними комбінаціями:
1. модель-1, датасет-1
2. модель-1, датасет-2
3. модель-2, датасет-2

Щоб зупинити контейнери Evidently виконуємо:
```
docker-compose down
```
