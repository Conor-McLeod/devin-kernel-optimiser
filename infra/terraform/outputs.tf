output "public_ip" {
  value = try(nebius_compute_v1_instance.h100.status.network_interfaces[0].public_ip_address.address, null)
}

output "ssh" {
  value = try("ssh -i ${trimsuffix(var.ssh_public_key_path, ".pub")} -L 8765:127.0.0.1:8765 ${var.username}@${split("/", nebius_compute_v1_instance.h100.status.network_interfaces[0].public_ip_address.address)[0]}", null)
}
