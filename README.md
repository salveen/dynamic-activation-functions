# Dynamic Activation Functions Playground

This repo demonstrates a lightweight experiment comparing a traditional
perceptron, fixed activations, and activation-aware neurons that can learn
their own thresholds. The focus is now on clarity over heavy design patterns:
activation behavior is embedded directly inside the `Neuron` class, and data
loading is streamlined around three real-world binary classification datasets.

## Datasets

| Key | Description | Notes |
| --- | ----------- | ----- |
| `breast_cancer` | Breast Cancer Wisconsin (from `sklearn`) | 30 numerical features, no extra prep required. |
| `titanic` | OpenML Titanic survival | Includes engineered categorical dummies for class, sex, and embarkation. |
| `heart_disease` | UCI Heart Disease (OpenML) | Cleans missing markers and converts diagnosis to binary. |

All datasets are standardized with `StandardScaler` and split with a
configurable `test_size` (default `0.2`).

## Running the experiment

```bash
python main.py
```

The script iterates through the three datasets, trains:

1. An `sklearn` Perceptron baseline.
2. A fixed ReLU neuron.
3. A dynamic ReLU neuron that learns slope/threshold.
4. A fixed step neuron.
5. An adaptive step neuron with a learnable threshold.

Results for every dataset/model pair are appended to `experiment_results.csv`.

## Customizing runs

To focus on a single dataset, adjust `datasets_to_test` inside `main.py` or
instantiate `PerceptronExperiment` manually:

```python
from data_utils import DatasetConfig
from main import PerceptronExperiment

config = DatasetConfig(dataset_type="titanic", test_size=0.25)
experiment = PerceptronExperiment(config, dataset_name="titanic")
experiment.run()
```

## Requirements

See `requirements.txt` for pinned versions (NumPy, pandas, scikit-learn).
Install them via `pip install -r requirements.txt` inside your virtualenv.
