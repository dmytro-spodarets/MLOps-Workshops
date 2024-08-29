# Майстерня з MLOps: AI від досліджень до деплою у продакшн

## День третій

### Робота з dstack

Створення gateway

`dstack apply -f gateway.dstack.yml`

`dstack gateway list`

DNS - add A DNS record for *.<gateway domain>

Деплой моделі

`dstack apply -f service-model.dstack.yml`

`curl -X POST "http://url:8000/ask" -H "Content-Type: application/json" -d '{"text": "What is DevOps?"}'`

Змінюємо на phi

`dstack apply -f service-model.dstack.yml`

Запускаэмо дев оточення

`dstack apply -f dev.dstack.yml`

Генерація документів

`python generate_devops_docs.py`

Робота з вектоную базою даних

- Реєструємось - https://qdrant.tech/
- Створення кластеру https://cloud.qdrant.io/accounts/c6e8d207-a875-4cf6-b662-21b2393ade6c/overview
- Завантаження даних - `python document_loader.py`
- Дашборд

Моніторинг - https://murnitur.ai/
- рейсінг
- Демонстріція детекції PII
