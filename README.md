## Overview over Datasets

Datasets are available in cleaned versions that can be fed into the TPOT algorithm. Own datasets need to be cleaned. 
(Missing values are okay, columns should have the right type, target variable needs to be known.)

- BankChurn dataset (Will a customer leave the credit card service or not?)
- Hypothyroid dataset (Does a patient have hypothyroid disease? If yes, which type?)
- Breastcancer dataset (Is the breast cancer malignant or benign?)
- Pokerhand dataset (deterministic classification of a set of cards corresponding to a pokerhand)

## TPOT

#### Installation

http://epistasislab.github.io/tpot/installing/

Install in anaconda (with pip: ```conda install pip```):
```conda install numpy scipy scikit-learn pandas joblib pytorch```
```pip install deap update_checker tqdm stopit xgboost```
```pip install dask[delayed] dask[dataframe] dask-ml fsspec>=0.3.3 distributed>=2.10.0```
```pip install tpot```
```pip install scikit-mdr skrebate```

#### Used versions:

- TPOT: 0.11.7
- xgboost (Gradient Boosting): 1.3.3
- scikit-klearn: 0.23.2
- scikit-mdr: 0.4.4
- scipy: 1.5.2
- pandas: 1.2.1
- numpy-base: 1.19.2
- dask/dask-core: 2021.1.1
- dask-glm: 0.2.0
- dask-ml: 1.8.0
- skrebate 0.61
- deap: 1.3.1
- joblib: 1.0.1

#### Run TPOT Algorithm executable to generate results:

Further explanation in ```TPOT_executable.py```.

- ```python TPOT_executable.py "bankchurners" "datasets/bankChurners/bc_cleaned.csv" "Attrition_Flag" 60```
- ```python TPOT_executable.py "hypothyroid" "datasets/hypothyroid/ht_cleaned.csv" "Class" 60```
- ```python TPOT_executable.py "breastcancer" "datasets/breastcancer/breast-cancer-diagnostic_cleaned.csv" "MALIGNANT" 60```
- ```python TPOT_executable.py "pokerhand" "datasets/pokerhand/pokerhand-normalized_cleaned.csv" "class" 60``` -> not working

#### For an overview, look into the ```Teach Talk TPOT.ipynb``` notebook.