resource "aws_apprunner_service" "ml_app_server" {
  service_name = var.name_ml_app_server

  source_configuration {
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ml_app_ecr_role.arn
    }

    image_repository {
      image_identifier      = "493395458839.dkr.ecr.us-east-1.amazonaws.com/linear-regression:latest"
      image_repository_type = "ECR"

      image_configuration {
        port = var.port_ml_app_server
        runtime_environment_variables = {
            WANDB_API_KEY = var.WANDB_API_KEY
            WANDB_CACHE_DIR = "/app/.cache"
        }
      }
    }
  }

  instance_configuration {
    cpu                = var.service_cpu
    memory             = var.service_memory
  }
}
