# Майстерня з MLOps: AI від досліджень до деплою у продакшн

## День другий

### Робота з dstack

`dstack init`

`dstack config --url http://127.0.0.1:3000 --project MLOps-workshop --token 724fe88a-eca8-4105-add4-697132b71f14`

`dstack fleet`

`dstack apply -f fleet.dstack.yml`

`dstack apply -f dev.dstack.yml`

`vscode://vscode-remote/ssh-remote+vscode/workflow`
`ssh vscode`

`dstack apply -f train.dstack.yml`


### Робота з Weights & Biases

Демо https://wandb.ai/
- Реєстрація - https://wandb.ai/site/pricing - Get Started
- API Key
- Локальне розгортання - https://docs.wandb.ai/quickstart
- Projects
- Model registry
- Додати нову версию моделі
