resource "aws_sagemaker_endpoint_configuration" "ml_app_endpoint_config" {
  name = "ml-app-endpoint-config"
  production_variants {
    variant_name          = "AllTraffic"
    model_name            = aws_sagemaker_model.ml_app_model.name
    initial_instance_count = 1
    instance_type         = "ml.m5.large"
  }
}

resource "aws_sagemaker_endpoint" "ml_app_endpoint" {
  name                         = "ml-app-endpoint"
  endpoint_config_name  = aws_sagemaker_endpoint_configuration.ml_app_endpoint_config.name
}
