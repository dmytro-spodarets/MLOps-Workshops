# Майстерня з MLOps: AI від досліджень до деплою у продакшн

## День перший - Робота з даними

### Необхідні пакети
```
pip install openai
```

### Створення власного датасету
Джерела Q&A:
* https://github.com/bregman-arie/devops-exercises/
* https://github.com/trimstray/test-your-sysadmin-skills

Джерела посилання:
* https://github.com/binhnguyennus/awesome-scalability
* https://github.com/upgundecha/howtheysre

Синтетичні дані - GPT-4o

Скрипти: 
* Генерація питань
```
python devops_qa_generator.py
```
* Формування датасету із джерел. Перед цим клонуемо необхідні репозиторії до `temp`
```
python generate_qa_dataset.py
```

Приклад нашого датасета
```json
[
  {
    "question": "What is DevOps?",
    "answer": "DevOps is a set of practices that combines software development (Dev) and IT operations (Ops). It aims to shorten the systems development life cycle and provide continuous delivery with high software quality. DevOps is complementary with Agile software development; several DevOps aspects came from Agile methodology."
  },
  {
    "question": "Explain the concept of continuous integration.",
    "answer": "Continuous Integration (CI) is a software development practice where developers regularly merge their code changes into a central repository, after which automated builds and tests are run. The key goals of CI are to find and address bugs quicker, improve software quality, and reduce the time it takes to validate and release new software updates."
  }
]
```

### Розмітка - Label Studio

Локальне розгортання використовуючи Docker
```
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

інші варіанти розгортання - https://labelstud.io/guide/install 

Web-інтерфейс - http://localhost:8080/

Q&A Шаблон

```
<View className="root">
  <Style>
  .root {
    font-family: 'Roboto', sans-serif;
    line-height: 1.6;
    background-color: #f0f0f0;
  }
  .container {
    margin: 0 auto;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 5px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.1);
  }
  .question {
    padding: 20px;
    background-color: #0084ff;
    color: #ffffff;
    border-radius: 5px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 3px 10px 0 rgba(0, 0, 0, 0.1);
  }
  .answer-input {
    flex-basis: 49%;
    padding: 20px;
    background-color: rgba(44, 62, 80, 0.9);
    color: #ffffff;
    border-radius: 5px;
    box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1), 0 3px 10px 0 rgba(0, 0, 0, 0.1);
    width: 100%;
    border: none;
    font-family: 'Roboto', sans-serif;
    font-size: 16px;
    outline: none;
  }
  .answer-input:focus {
    outline: none;
  }
  .answer-input:hover {
    background-color: rgba(52, 73, 94, 0.9);
    cursor: pointer;
    transition: all 0.3s ease;
  }
  .lsf-richtext__line:hover {
    background: unset;
  }
  </Style>
  <View className="container">
    <View className="question">
      <Text name="question" value="$question"/>
    </View>
    <View className="answer-input">
      <TextArea name="answer" toName="question" maxSubmissions="1" required="true" requiredMessage="You must provide the response to the question" placeholder="Type your answer here..." rows="10"/>
    </View>
  </View>
</View><!-- {"data" : {
  "question": "Can you explain the key differences between continuous integration, continuous delivery, and continuous deployment?"
}}
-->
```

### Версіювання - DVC
Створюємо репозиторій для нашого датасету - devops-qa-dataset

Клонуємо репозиторій
```
git clone git@github.com:dmytro-spodarets/devops-qa-dataset.git
```

Створюємо віртуальне середовище Python
```
python -m venv .venv
source .venv/bin/activate
```

Встановлюємо DVC
```
pip install "dvc[s3]"
```
Ініціюємо DVC
```
dvc init
```
Cтворюємо .gitignore та додаємо до нього .venv

Робимо перший коміт
```
git add .gitignore
git commit -m "Initialize DVC"
```

Копіюємо наш датасет
```
cp ../MLOps-Workshops/From_Research_to_Production/Day-1/devops_qa_dataset.json ./devops_qa_dataset.json
```

Додаємо файл для трекінгу
```
dvc add devops_qa_dataset.json
```

комітемо першу версію датасета
```
git add devops_qa_dataset.json.dvc .gitignore
git commit -m "DevOps V1"
```

Підключаємо віддалений сторедж
```
dvc remote add -d storage s3://devops-qa-dataset
```

Завантажуємо дані на S3
```
dvc push
```
Комітемо зміну у конфігурації dvc
```
git add .dvc/config
git commit -m "add remote storage"
```
Копіюємо новий датасет та створюємо наступну версію
```
cp ../MLOps-Workshops/From_Research_to_Production/Day-1/devops_qa_dataset_new.json ./devops_qa_dataset_new.json
dvc add devops_qa_dataset_new.json
git add devops_qa_dataset_new.json.dvc .gitignore
git commit -m "DevOps V2"
```

Робимо доразмітку, копіюємо файл та та створюємо наступну версію датасета
```
cp ../MLOps-Workshops/From_Research_to_Production/Day-1/devops_qa_dataset_new.json ./devops_qa_dataset_new.json
dvc status
dvc add devops_qa_dataset_new.json
git add devops_qa_dataset_new.json.dvc 
git commit -m "DevOps V3"
dvc push
```

Якщо ми хочемо завантажити данні, тоді ми робимо
```
git clone ...
dvc pull
```
Якщо ми хочему якусь конкретну версію, тоді ми робимо
```
git checkout <...>
dvc checkout
```
