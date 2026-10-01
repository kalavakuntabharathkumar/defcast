# Azure ML deployment outline

- Create an Azure ML workspace and managed identity in your own subscription.
- Register the reviewed dataset and version the data asset; do not use the generated toy images for business claims.
- Package a pinned environment and training entry point, submit a reproducible command job, and log metrics/model artifacts.
- Evaluate by product line, camera, lighting, and defect type; inspect recall/precision and calibration before deployment.
- Register only approved models, deploy behind authenticated endpoints, monitor drift and latency, and document rollback criteria.
- GitHub Actions should use federated credentials or a secret store; do not put Azure credentials in repository files.
