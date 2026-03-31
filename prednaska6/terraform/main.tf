terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = "project-99ae20df-3cbc-4d1d-9e1"
  region  = "europe-central2"
  zone    = "europe-central2-a"
}

resource "google_compute_instance" "default" {
  name         = "uui-iris-predictor-vm"
  machine_type = "e2-micro"

  tags = ["allow-web-access"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"

    access_config {}
  }

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_google.pub")}"
  }

}

resource "google_compute_firewall" "allow_ports" {
  name    = "allow-specific-ports"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22", "8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["allow-web-access"]
}
