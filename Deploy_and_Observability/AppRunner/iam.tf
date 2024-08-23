resource "aws_iam_role" "apprunner_ml_app_ecr_role" {
  name = "apprunner-ecr-role-ml-app-server"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ml_app_ecr_policy_attachment" {
  role       = aws_iam_role.apprunner_ml_app_ecr_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}
