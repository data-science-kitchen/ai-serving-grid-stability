# Industry Challenge: AI Serving Grid Stability

## Anomaly Detection in the European Transmission Grid for Electricity

Europeans rarely experience blackouts due to a reliable electricity supply, maintained by companies like TransnetBW, one of Germany's four transmission system operators. They ensure grid stability by balancing electricity input and output to maintain a frequency of 50 Hertz, using automated signals to adjust power plant activity. Traditionally, each European state managed its own grid, often leading to inefficiencies. Since 2022, the PICASSO platform, operated by TransnetBW, has optimized this process across Europe, saving hundreds of millions of euros annually and aiding decarbonization. To ensure platform reliability, TransnetBW monitors various time series data, offering a new dataset for further research.

Challenge Website: https://hessian.ai/industry-challenge/

Kaggle competition: https://www.kaggle.com/competitions/ai-serving-grid-stability/overview

## Getting Started

This guide is divided into two sections: **Running `submission.py`** and **Notebooks**. The first section explains how to run our code to generate our best submission, while the second section covers testing using notebooks. To reproduce our results, follow the instructions for running `submission.py`.

### Running `submission.py`

The `submission.py` script contains the code needed to generate the `submission.csv` file. To reproduce our results, follow these steps:

1. git clone https://github.com/data-science-kitchen/ai-serving-grid-stability.git
2. cd ai-serving-grid-stability
3. `py -m venv venv` (Python 3.9.1) or replace `py` with your installed Python version.
4. Activate your venv
5. `pip install requirements.txt`
6. Download [train.csv](https://www.kaggle.com/competitions/ai-serving-grid-stability/data?select=train.csv) from Kaggle.
7. Put the `train.csv` and `test.csv` into the data folder.
8. py submission.py

### Install Notebook

Just test notebooks.

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
