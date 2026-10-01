# Defect Detection and Demand Forecasting Pipeline with MLOps

Portfolio-ready starter that demonstrates image data validation, a PyTorch CNN training entry point, optional Stable Diffusion augmentation interface, scikit-learn demand forecasting, experiment metadata, and CI tests. No pretrained weights, proprietary dataset, or cloud resources are bundled.

## Setup (Windows first)

Python 3.10–3.12 is recommended. Create a virtual environment and install the baseline:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pipeline.make_demo_data
python -m pipeline.forecast
pytest -q
```

For CNN training, install a compatible PyTorch build from the official PyTorch selector, then run `python -m pipeline.train_cnn`. The demo image set is generated geometric data only and is not representative of real industrial defects. The baseline forecast uses generated demand values.

## Pipeline

1. `pipeline/data_validation.py` validates image paths, labels, and tabular input schema.
2. `pipeline/make_demo_data.py` creates a small, clearly labelled toy dataset for smoke tests.
3. `pipeline/train_cnn.py` trains a compact CNN and logs metrics; the script fails with an actionable message if PyTorch is not installed.
4. `pipeline/augmentation.py` defines a safe optional hook for externally generated, reviewed synthetic images. It deliberately does not fabricate Stable Diffusion outputs or silently mix synthetic and real labels.
5. `pipeline/forecast.py` fits a scikit-learn random-forest regressor and reports MAPE.
6. `pipeline/experiment.py` logs locally and optionally to MLflow.
7. `.github/workflows/ci.yml` runs validation and tests on pushes and pull requests.

Azure ML deployment is documented in `infra/azure-ml.md`; no cloud account or resources are created by running this repository. Metrics in a resume must come from experiments on a documented, labelled dataset—not from this toy demo.
