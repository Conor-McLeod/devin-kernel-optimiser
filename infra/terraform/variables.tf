variable "project_id" {
  description = "Nebius project ID (parent_id). Must be a project in a region with H100s, e.g. eu-north1."
  type        = string
}

variable "subnet_name" {
  description = "Existing subnet in the project (see `nebius vpc subnet list`)."
  type        = string
}

variable "name" {
  type    = string
  default = "devin-h100"
}

variable "preset" {
  description = "1gpu-16vcpu-200gb (1x H100) or 8gpu-128vcpu-1600gb (8x H100, one GPU per agent)."
  type        = string
  default     = "1gpu-16vcpu-200gb"
}

variable "image_family" {
  # Must match the CUDA major version torch is built for. sglang (0.5.23) installs
  # torch 2.14 +cu130, which needs driver 580 (the cuda12 image ships 570 and fails with
  # "NVIDIA driver too old"). This image also has nvcc 13.0, so torch's extension
  # builder (Marlin JIT, CUDA C++ agents) gets a matching compiler.
  type    = string
  default = "ubuntu24.04-cuda13.0"
}

variable "disk_gib" {
  description = "Room for two venvs (sglang, vllm), CUTLASS, model configs and agent clones."
  type        = number
  default     = 300
}

variable "username" {
  description = "Login user created by cloud-init (not root/admin)."
  type        = string
  default     = "hack"
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "repo_url" {
  description = "Optional: public HTTPS URL of this repo, cloned to /opt/hackathon/main at boot. Leave empty for a private repo and clone it yourself."
  type        = string
  default     = ""
}

variable "stopped" {
  description = "Set true and apply to stop the VM (stops GPU billing; disk and static IP are kept)."
  type        = bool
  default     = false
}
