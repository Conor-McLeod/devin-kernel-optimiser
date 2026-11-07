# One Nebius H100 VM for the hackathon.
#
#   export NEBIUS_IAM_TOKEN=$(nebius iam get-access-token)   # valid 12h
#   cp terraform.tfvars.example terraform.tfvars && $EDITOR terraform.tfvars
#   terraform init && terraform apply
#
# Cost: ~$4.50/GPU-hour on demand (nebius.com/prices). Set `stopped = true` and
# apply to pause overnight, `terraform destroy` when the hackathon is over.

terraform {
  required_version = ">= 1.5"
  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.8"
    }
  }
}

# Auth comes from NEBIUS_IAM_TOKEN (user account). For a service account, see
# https://docs.nebius.com/terraform-provider/authentication
provider "nebius" {}

data "nebius_vpc_v1_subnet" "this" {
  name      = var.subnet_name
  parent_id = var.project_id
}

resource "nebius_compute_v1_instance" "h100" {
  name      = var.name
  parent_id = var.project_id
  stopped   = var.stopped

  resources = {
    platform = "gpu-h100-sxm"
    preset   = var.preset
  }

  boot_disk = {
    attach_mode = "READ_WRITE"
    managed_disk = {
      name = "${var.name}-boot"
      spec = {
        type           = "NETWORK_SSD"
        size_gibibytes = var.disk_gib
        source_image_family = {
          image_family = var.image_family
        }
      }
    }
  }

  network_interfaces = [
    {
      name              = "eth0"
      ip_address        = {}
      public_ip_address = { static = true } # survives stop/start
      subnet_id         = data.nebius_vpc_v1_subnet.this.id
    }
  ]

  cloud_init_user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    username       = var.username
    ssh_public_key = trimspace(file(pathexpand(var.ssh_public_key_path)))
    repo_url       = var.repo_url
  })
}
