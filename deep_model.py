import deepchem as dc
import numpy as np
from deepchem.models import DMPNNModel
from deepchem.splits import RandomSplitter
from deepchem.hyper import RandomHyperparamOpt
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(run_kfold=True, csv_file="desalted_BioHC.csv", tasks=None):
    """Main function to run the DMPNN model training and evaluation.
    
    Args:
        run_kfold (bool): Whether to run K-fold cross-validation. Default is True.
        csv_file (str): Path to the CSV file containing the dataset. Default is "desalted_BioHC.csv".
        tasks (list): List of task names to predict. Default is ["LogHalfLife"].
    """
    
    if tasks is None or len(tasks) == 0:
        raise ValueError("Error: No tasks specified. Please provide at least one task name using --tasks argument.")
    
    print(f"Loading dataset from: {csv_file}")
    print(f"Tasks: {tasks}")
    
    # Load custom dataset from CSV

    # Initialize the loader for graph-based models
    loader = dc.data.CSVLoader(tasks=tasks, feature_field="SMILES", featurizer=dc.feat.DMPNNFeaturizer())
    dataset = loader.featurize(csv_file)

    # Split the train+val and test dataset (80% train+val, 20% test)
    splitter = RandomSplitter()
    train_val_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8)

    # Split the train and validation dataset (70% train, 10% validation)
    train_dataset, valid_dataset = splitter.train_test_split(train_val_dataset, frac_train=7/8)

    # Define metrics
    pearson_r2_score = dc.metrics.Metric(dc.metrics.pearson_r2_score)
    mae_error = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    rms_score = dc.metrics.Metric(dc.metrics.rms_score)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # K-Fold Cross-Validation (optional)
    if run_kfold:
        print("\n" + "="*50)
        print("K-Fold Cross-Validation")
        print("="*50)
        
        num_folds = 5
        # KFold splits indices based on the number of samples in the dataset
        train_val_dataset_folds = splitter.k_fold_split(train_val_dataset, k=num_folds)

        # Collect metrics
        k_r2_scores = []
        k_rmse_scores = []
        k_mae_scores = []

        # Cross-validation loop
        for fold, (k_train_dataset, k_valid_dataset) in enumerate(train_val_dataset_folds):
            print(f"\nFold {fold+1}/{num_folds}")

            # Re-instantiate model for each fold
            model = DMPNNModel(n_tasks=len(tasks), mode='regression', learning_rate=1e-3, batch_size=64)

            # Train model
            model.fit(k_train_dataset, nb_epoch=100)

            # Evaluate
            scores = model.evaluate(k_valid_dataset, [pearson_r2_score, mae_error, rms_score])

            k_r2_scores.append(scores['pearson_r2_score'])
            k_rmse_scores.append(scores['rms_score'])
            k_mae_scores.append(scores['mean_absolute_error'])

        # Report mean ± std for each metric
        print("\n=== Cross-Validation Summary ===")
        print(f"R²     : {np.mean(k_r2_scores):.4f} ± {np.std(k_r2_scores):.4f}")
        print(f"RMSE   : {np.mean(k_rmse_scores):.4f} ± {np.std(k_rmse_scores):.4f}")
        print(f"MAE    : {np.mean(k_mae_scores):.4f} ± {np.std(k_mae_scores):.4f}")
    else:
        print("\nSkipping K-Fold Cross-Validation...")

    # Hyper-Parameter Tuning
    print("\n" + "="*50)
    print("Hyper-Parameter Tuning")
    print("="*50)

    # Define the model builder function for DMPNN
    def model_builder(model_dir, **model_params):
        return DMPNNModel(
            n_tasks=len(tasks),
            mode='regression',
            model_dir=model_dir,
            **model_params
        )

    # Define hyperparameter space for DMPNN
    param_dict = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [32, 64, 128],
        'global_features_size': [0],
        'enc_hidden': [60, 150, 300],
        'depth': [2, 3, 4],
        'enc_dropout_p': [0.0, 0.2, 0.4],
        'aggregation': ['mean'],
        'aggregation_norm': [100],
        'ffn_hidden': [60, 150, 300],
        'ffn_layers': [2, 3],
        'ffn_dropout_p': [0.0, 0.2, 0.4],
        'ffn_dropout_at_input_no_act': [True, False],
    }

    # Set up and run random search
    search = RandomHyperparamOpt(model_builder=model_builder, max_iter=10)  # change max_iter to 60 for full random search
    best_model, best_params, all_results = search.hyperparam_search(
        param_dict, train_dataset, valid_dataset, pearson_r2_score, 
        nb_epoch=100, use_max=True, logdir="./random_search"
    )

    # Evaluate best model
    print("Validation Set Metrics for best model")
    val_scores = best_model.evaluate(valid_dataset, [pearson_r2_score, mae_error, rms_score])
    print(val_scores)

    # 3 random seed models on best parameters
    print("\n" + "="*50)
    print("Testing with 3 random seeds on best parameters")
    print("="*50)

    # Remove model_dir from best_params for reusing
    if 'model_dir' in best_params:
        del best_params['model_dir']

    # Collect metrics
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for i in range(3):
        print(f"\nTraining model {i+1}/3...")
        model = DMPNNModel(
            n_tasks=len(tasks),
            mode='regression',
            **best_params
        )
        model.fit(train_dataset, nb_epoch=100)
        
        # Evaluate on test
        scores = model.evaluate(test_dataset, [pearson_r2_score, mae_error, rms_score])

        r2_scores.append(scores['pearson_r2_score'])
        rmse_scores.append(scores['rms_score'])
        mae_scores.append(scores['mean_absolute_error'])

    # Report mean ± std for each metric
    print("\n=== Test Results Summary ===")
    print(f"R²     : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"RMSE   : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"MAE    : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")

    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DMPNN Model Training and Evaluation')
    parser.add_argument('--no-kfold', action='store_true', 
                       help='Skip K-fold cross-validation (default: run K-fold)')
    parser.add_argument('--csv-file', type=str, default='desalted_BioHC.csv',
                       help='Path to CSV file containing the dataset (default: desalted_BioHC.csv)')
    parser.add_argument('--tasks', nargs='+', default=['LogHalfLife'],
                       help='List of task names to predict (default: LogHalfLife)')
    
    args = parser.parse_args()
    
    # Run main function with specified parameters
    main(run_kfold=not args.no_kfold, csv_file=args.csv_file, tasks=args.tasks)
