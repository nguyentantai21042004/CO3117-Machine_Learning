# Đánh giá mô hình

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # type: ignore
from src.utils import logger

def evaluate_model_performance(y_test, y_pred):
    """
    Evaluate model performance using various metrics.
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        logger.info("Calculating model performance metrics...")
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log results
        logger.info(f"Mean Squared Error (MSE): {mse:.6f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.6f}")
        logger.info(f"R-squared (R²): {r2:.6f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        raise

def plot_predictions(y_test, y_pred, output_dir='../assignment/evaluate_results'):
    """
    Plot actual vs predicted values.
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save the plot
    """
    try:
        logger.info("Plotting actual vs predicted values...")
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot actual and predicted values
        plt.plot(y_test.values, label='Actual CO Concentration', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted CO Concentration', color='red', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Sample Index')
        plt.ylabel('CO(GT) Concentration')
        plt.title('Actual vs Predicted CO Concentration')
        plt.legend()
        plt.grid(True)
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = f'{output_dir}/predictions_comparison.png'
        plt.savefig(output_path)
        logger.info(f"Predictions plot saved to '{output_path}'")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        raise

def plot_residuals(y_test, y_pred, output_dir='../assignment/evaluate_results'):
    """
    Plot residuals (differences between actual and predicted values).
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        output_dir (str): Directory to save the plot
    """
    try:
        logger.info("Plotting residuals...")
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot residuals
        plt.scatter(y_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        
        # Add labels and title
        plt.xlabel('Actual CO Concentration')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True)
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = f'{output_dir}/residuals_plot.png'
        plt.savefig(output_path)
        logger.info(f"Residuals plot saved to '{output_path}'")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting residuals: {str(e)}")
        raise

def save_evaluation_results(metrics, output_dir='../assignment/evaluate_results'):
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (str): Directory to save the results
    """
    try:
        logger.info("Saving evaluation results...")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert metrics to DataFrame and save
        metrics_df = pd.DataFrame([metrics])
        output_path = f'{output_dir}/evaluation_metrics.csv'
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Evaluation metrics saved to '{output_path}'")
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")
        raise

def main():
    """
    Main function to execute the evaluation pipeline.
    """
    try:
        logger.info("Starting model evaluation pipeline...")
        
        # Load test data and predictions
        y_test = pd.read_csv('../dataset/split_data/y_test.csv')['CO(GT)']
        y_pred = pd.read_csv('../assignment/train_results/predictions.csv')['Predicted']
        
        # Evaluate model performance
        metrics = evaluate_model_performance(y_test, y_pred)
        
        # Create visualizations
        plot_predictions(y_test, y_pred)
        plot_residuals(y_test, y_pred)
        
        # Save results
        save_evaluation_results(metrics)
        
        logger.info("Model evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise