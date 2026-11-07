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
  # cuda12 (toolkit 12.8), not the recommended cuda13.0: torch wheels are built for CUDA 12.x
  # and torch's extension builder refuses an nvcc with a different major version, which
  # would break Marlin's JIT build and the CUDA C++ agents.
  type    = string
  default = "ubuntu24.04-cuda12"
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
