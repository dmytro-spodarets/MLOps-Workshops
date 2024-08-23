

variable "port_ml_app_server" {
  type        = number
  default     = 5000
  description = ""
}

variable "WANDB_API_KEY" {
  description = "WANDB API KEY"
  type        = string
  sensitive   = true
}

variable "aws_region" {
  description = "(Optional) AWS Region."
  type = string
  default = "us-east-1"
}