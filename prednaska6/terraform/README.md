## Prerequisites

* [Terraform](https://developer.hashicorp.com/terraform/downloads)
* [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install)
* A Google Cloud account
* A GCP project created

## Create SSH key

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

## Authenticate with Google Cloud

Log in to Google Cloud with:

```
gcloud auth application-default login
```

## Initialize Terraform

Initialize Terraform:

```
terraform init
```

## Apply the Configuration

```
terraform apply
```

Type `yes` when prompted.

## Access the VM

After successful deployment, Terraform will output the VM details.

To SSH into the VM:

```
ssh -i path_to_ssh_key ubuntu@vm_ip_address
```

## Destroy Resources (Cleanup)

To delete the created infrastructure:

```
terraform destroy
```
