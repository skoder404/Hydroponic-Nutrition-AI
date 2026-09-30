# Hydroponic Nutrition AI

This project trains a machine learning model to classify hydroponic lettuce conditions from sensor readings such as pH, EC, water temperature, humidity, and air temperature.

## Project structure

- `sensor_model_training.ipynb` — main notebook for data loading, model comparison, tuning, evaluation, and saving the final model.
- `Hydroponic Lettuce Dataset/` — training, test, and full datasets used for experimentation.
- `outputs/` — generated artifacts such as confusion matrix and saved model. This folder is ignored by Git.
- `mlm/` — local virtual environment. This folder is ignored by Git.

## Dataset

The dataset is stored under `Hydroponic Lettuce Dataset/` and includes:

- `hydroponic_lettuce_train.csv`
- `hydroponic_lettuce_test.csv`
- `hydroponic_lettuce_full.csv`

## Model

The notebook trains a tuned Random Forest classifier and saves the final pipeline to:

- `outputs/hydro_sensor_model.joblib`

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv mlm
source mlm/bin/activate   # Linux/macOS
mlm\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

## Run the notebook

Open `sensor_model_training.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

## Notes

The project intentionally uses grouped cross-validation by date to avoid leakage between related sensor readings.
