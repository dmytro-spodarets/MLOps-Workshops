terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.11.0"
    }

  }

  required_version = ">= 1.5.4"

backend "local" {}
}

provider "aws" {
  region  = var.aws_region
}