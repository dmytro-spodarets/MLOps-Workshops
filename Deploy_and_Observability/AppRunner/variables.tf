variable "name_ml_app_server" {
  description = ""
  type = string
  default = "ml-api"
}

variable "service_cpu" {
  type        = number
  default     = 1024
  description = "The number of CPU units reserved for container."
}

variable "service_memory" {
  type        = number
  default     = 2048
  description = "The amount (in MiB) of memory reserved for container."
}

variable "port_ml_app_server" {
  type        = number
  default     = 8080
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