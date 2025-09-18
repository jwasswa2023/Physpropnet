# Physpropnet

A molecular property prediction toolkit using DeepChem with both deep learning (DMPNN) and traditional machine learning (Random Forest) approaches.

## Overview

This repository contains two main scripts for molecular property prediction:

- **`deep_model.py`**: Uses DMPNN (Directed Message Passing Neural Network) for deep learning-based prediction
- **`new_shallow_model.py`**: Uses Random Forest Regressor with RDKit descriptors for traditional ML-based prediction

## Requirements

- Python 3.7+
- DeepChem
- PyTorch
- scikit-learn
- RDKit
- numpy

## Dataset Format

Your CSV file should contain:
- A `SMILES` column with molecular structures
- A column with the target property you want to predict

Example:
```csv
SMILES,LogHalfLife
CCO,1.2
CC(C)O,1.8
...
```

## Usage

### Deep Model (DMPNN)

The deep model script uses graph neural networks for molecular property prediction.

#### Basic Usage
```bash
python deep_model.py
```

#### Custom Dataset and Property
```bash
# Use custom dataset and property
python deep_model.py --csv-file "desalted_BioHC.csv" --tasks "LogHalfLife"

# Skip K-fold cross-validation for faster execution
python deep_model.py --csv-file "desalted_BioHC.csv" --tasks "LogHalfLife" --no-kfold
```

#### Command-line Arguments
- `--csv-file`: Path to CSV file containing the dataset (default: desalted_BioHC.csv)
- `--tasks`: Task name to predict (default: LogHalfLife)
- `--no-kfold`: Skip K-fold cross-validation (default: run K-fold)
- `--help`: Show help message

### Shallow Model (Random Forest)

The shallow model script uses traditional machine learning with molecular descriptors.

#### Basic Usage
```bash
python new_shallow_model.py
```

#### Custom Dataset and Property
```bash
# Use custom dataset and property
python new_shallow_model.py --csv-file "desalted_BioHC.csv" --tasks "LogHalfLife"

# Skip K-fold cross-validation for faster execution
python new_shallow_model.py --csv-file "desalted_BioHC.csv" --tasks "LogHalfLife" --no-kfold
```

#### Command-line Arguments
- `--csv-file`: Path to CSV file containing the dataset (default: desalted_BioHC.csv)
- `--tasks`: Task name to predict (default: LogHalfLife)
- `--no-kfold`: Skip K-fold cross-validation (default: run K-fold)
- `--help`: Show help message

## Procedure

### Dataset Splitting
The dataset was initially divided into a training set and a held-out test set. The held-out test set was reserved solely for the final evaluation of model performance, ensuring an unbiased estimate of generalization.

### K-Fold Benchmarking on Training Set
To understand the internal variability of model performance, the training set was subjected to k-fold cross-validation using default hyperparameters. This benchmarking allowed for an initial assessment of how model performance varied across different folds and provided insights into model stability.

### Training-Validation Split
Given that the standard deviation of performance across the k-folds was low, indicating relatively stable performance, and considering computational resource limitations, we opted for a simpler approach. The training set was split into a new training subset and a validation subset. This enabled faster iterative experimentation while still maintaining a reliable evaluation set for hyperparameter tuning.

### Hyperparameter Optimization
A hyperparameter search consisting of 60 trials was performed using the new training subset. Each trial involved training a model with a specific hyperparameter configuration and evaluating its performance on the validation subset. The hyperparameter set that achieved the best validation performance was selected for final model training.

### Final Model Training and Evaluation
Using the best-found hyperparameters, three models were independently trained on the entire original training set with different random seeds to account for variability due to initialization and stochasticity in training. Each model was then evaluated on the held-out test set. The mean performance and standard deviation across these three models were reported to provide a robust estimate of model performance and its variability.

## Output

Both scripts provide:
1. **K-fold Cross-Validation Results** (optional): Mean ± std for R², RMSE, and MAE
2. **Hyperparameter Tuning**: Best parameters found through random search
3. **Final Test Results**: Performance on test set with 3 different random seeds
4. **Best Hyperparameters**: Complete list of optimal parameters

## Examples

### Quick Test (Skip K-fold)
```bash
python deep_model.py --no-kfold
python new_shallow_model.py --no-kfold
```

### Custom Dataset and Property
```bash
python deep_model.py --csv-file "my_molecules.csv" --tasks "Property1"
python new_shallow_model.py --csv-file "my_molecules.csv" --tasks "Property1"
```

## Notes

- Both scripts automatically handle train/validation/test splits (70%/10%/20%)
- Hyperparameter tuning uses random search with configurable iterations
- Results are saved to `./random_search` (deep) and `./random_search_shallow` (shallow) directories
- All models are evaluated using R², RMSE, and MAE metrics