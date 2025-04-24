import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from src.utils import logger

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    """
    Train a Random Forest model with specified parameters.
    
    Args:
        X_train (pd.DataFrame): Training features (excluding CO(GT))
        y_train (pd.Series): Training target (CO(GT))
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum number of samples required to split a node
        random_state (int): Random seed for reproducibility
        
    Returns:
        RandomForestRegressor: Trained model
    """
    try:
        logger.info("Initializing Random Forest model...")
        logger.info(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        
        # Initialize model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        # Train model
        logger.info("Training Random Forest model...")
        rf_model.fit(X_train, y_train)
        logger.info("Model training completed successfully!")
        
        return rf_model
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using test data.
    
    Args:
        model (RandomForestRegressor): Trained model
        X_test (pd.DataFrame): Test features (excluding CO(GT))
        y_test (pd.Series): Test target (CO(GT))
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Log results
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"R-squared (R²): {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance from the trained model.
    
    Args:
        model (RandomForestRegressor): Trained model
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
    """
    try:
        logger.info("Plotting feature importance...")
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(min(top_n, len(sorted_features))), 
                sorted_importances[:top_n], 
                align="center")
        plt.yticks(range(min(top_n, len(sorted_features))), 
                  sorted_features[:top_n])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs('../assignment/train_results', exist_ok=True)
        
        # Save plot
        output_path = '../assignment/train_results/feature_importance.png'
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to '{output_path}'")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def save_model_results(model, evaluation_metrics, output_dir='../assignment/train_results'):
    """
    Save model results and metrics to files.
    
    Args:
        model (RandomForestRegressor): Trained model
        evaluation_metrics (dict): Dictionary containing evaluation metrics
        output_dir (str): Directory to save results
    """
    try:
        logger.info("Saving model results...")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([evaluation_metrics])
        metrics_df.to_csv(f'{output_dir}/model_metrics.csv', index=False)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': evaluation_metrics['y_test'],
            'Predicted': evaluation_metrics['y_pred']
        })
        predictions_df.to_csv(f'{output_dir}/predictions.csv', index=False)
        
        logger.info(f"Model results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving model results: {str(e)}")
        raise

