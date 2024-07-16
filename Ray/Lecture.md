# Майстерня з MLOps: використання Ray

## Розгортання та налаштування Ray Cluster

Встановлення Ray
```
pip install -U "ray[all]"
```
### AWS
Конфігурування кластера відбуваеться через файл `cluster-config.yaml`, який далі використовується для його запуска:
```
ray up -y cluster-config-aws.yaml
```
Приклад [cluster-config-aws.yaml](cluster-config-aws.yaml) для розгортяння в AWS

https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html

### Kubernetes
Розгорніть оператор KubeRay із Helm chart репозиторія
```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1
kubectl get pods
```

Розгорніть ресурси RayCluster
```
helm install raycluster kuberay/ray-cluster --version 1.1.1 --set 'image.tag=2.9.0-aarch64'
kubectl get rayclusters
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay
```

Кастомізація кластеру - https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html

Отримати інформацію про головну ноду 
```
kubectl get service raycluster-kuberay-head-svc
```

Порт форвардінг для доступу до Ray Dashboard
```
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
```

Вхід до Ray Dashboard
```
http://localhost:8265
```

Запуск тестової задачі
```
ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

Видалення RayCluster
```
helm uninstall raycluster
kubectl get pods
helm uninstall kuberay-operator
kubectl get pods
```

## Навчання моделі
```
ray job submit --address http://localhost:8265 --working-dir . -- python run_training.py
```

## Оптимізація гіперпараметрів
```
ray job submit --address http://localhost:8265 --working-dir . -- python tune_hyperparameters.py
```

## Сервінг моделей
https://docs.ray.io/en/latest/serve/configure-serve-deployment.html
```
serve run object_detection:entrypoint
RAY_ADDRESS='http://localhost:8265' ray job submit --no-wait --working-dir . -- sh deploy_script.sh
```

## Різні команди роботи з джобами:
```
ray job logs JOB_ID
ray job status JOB_ID
ray job stop JOB_ID
ray job delete JOB_ID
ray job list
```
https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html