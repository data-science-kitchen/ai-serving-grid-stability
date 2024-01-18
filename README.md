# Industry Challenge: AI Serving Grid Stability
#### Anomaly Detection in the European Transmission Grid for Electricity

Challenge Website: https://hessian.ai/industry-challenge/

Kaggle competition: https://www.kaggle.com/competitions/ai-serving-grid-stability/overview

## Getting started

### Install Notebook

1. `py -m venv venv` (Python 3.9.1) or replace `py` with your installed Python version.
2. Activate your venv
3. `pip install requirements.txt`
4. Add Kernel: `python -m ipykernel install --user --name ai-serving-grid-stab --display-name "Ai Serving Grid Stability"` (Restart VSCode)
5. Select your kernel in your Notebook.


### Get the data
1. Install Kaggle API ([see instructions](https://www.kaggle.com/docs/api))
2. Run ```kaggle competitions download -c ai-serving-grid-stability```
3. Extract zip-file to `data` subdirectory: ```unzip -d data ai-serving-grid-stability.zip```

### Submission via Terminal

1. `kaggle competitions submit -c ai-serving-grid-stability -f submission.csv -m "MESSAGE"`
