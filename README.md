# Imputation and Semi Supervised Learning

A missing data imputation shootout plus semi supervised label propagation: three imputation strategies battle on apple quality data, then LabelSpreading recovers destroyed mushroom labels well enough for an SVC to hit 0.92 accuracy.

Coursework from **Applied Unsupervised Learning in Python** (University of Michigan, More Applied Data Science with Python specialization). Both notebooks run end to end with outputs.

## Data

`data/` holds the course provided CSVs:

* `apple_quality.csv`: 100 apple records used here (7 numeric features: Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity; target Quality good/bad)
* `mushrooms.csv`: 100 mushroom records used here (22 categorical features; target class edible/poisonous)

## What was done

**Part 1: Imputation shootout** (`notebooks/01-imputation-shootout.ipynb`)
Erased 70 percent of the feature cells at random (seed 42), then compared three strategies with 7 fold cross validation R² on a RandomForestRegressor: fill with zero plus a missingness indicator (0.1487), drop rows with any NaN keeping only 30 rows (0.1548), and kNN imputation plus indicator (0.2222, the winner). Full data baseline: 0.2882.

**Part 2: Label propagation** (`notebooks/02-label-propagation.ipynb`)
Built a 100 by 100 Gower distance matrix over the categorical mushroom features, split train/test, then destroyed 70 percent of the training labels per class (only 20 of 75 labels survived). LabelSpreading (kNN kernel, 9 neighbors) inferred the missing labels, and an SVC trained on the inferred labels scored 0.92 accuracy on the test set. Confusion matrix included: 2 of the 7 poisonous mushrooms were misclassified as edible.

## Tech

Python, pandas, NumPy, scikit learn (SimpleImputer, KNNImputer, LabelSpreading, SVC), matplotlib, Jupyter.
