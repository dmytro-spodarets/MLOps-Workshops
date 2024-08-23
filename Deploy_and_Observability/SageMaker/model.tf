resource "aws_sagemaker_model" "ml_app_model" {
  name               = "ml-app-sagemaker-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn
  primary_container {
    image           = "493395458839.dkr.ecr.us-east-1.amazonaws.com/linear-regression:latest"
    mode            = "SingleModel"
    container_hostname = "ml-app-container"
    environment = {
      WANDB_API_KEY = var.WANDB_API_KEY
      WANDB_CACHE_DIR = "/app/.cache"
    }
  }
}
