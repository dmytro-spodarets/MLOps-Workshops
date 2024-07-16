# MLOps-Workshops

Налаштування

Використання віртуального середовища Python
```
python -m venv .venv
source .venv/bin/activate
```

Налаштування AWS
```
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
aws configure
pip install boto3
```


Створення локального Kubernetes кластеру
```
https://www.docker.com/products/docker-desktop/
brew install helm
brew install kubectl or brew install kubectl
brew install minikube

minikube start --driver=docker
```

https://k8slens.dev/

Видалення локального Kubernetes кластеру
```
minikube delete --all
```