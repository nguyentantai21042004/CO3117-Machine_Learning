import pandas as pd # type: ignore
from src.data_preprocessing import preprocess_data
from src.data_splits import split_data
from src.utils import logger    
from src.train import train_random_forest, evaluate_model, plot_feature_importance, save_model_results  
from src.evaluate import evaluate_model_performance, plot_predictions, plot_residuals, save_evaluation_results

def main():
    logger.info("===================================================================")
    logger.info("=================Starting the main function...=====================")
    logger.info("===================================================================")
    dataset = pd.read_csv("dataset/AirQualityUCI.csv", delimiter=";")   

    logger.info("=================Preprocessing the dataset...=====================")
    dataset = preprocess_data(dataset)
    logger.info("Data preprocessing completed successfully!")
    logger.info("===================================================================")
    
    logger.info("=================Splitting the dataset...=====================")
    X_train, X_test, y_train, y_test, scaler = split_data(dataset)
    logger.info("Data splitting completed successfully!")
    logger.info("===================================================================")
    
    # Train model
    model = train_random_forest(
        X_train, y_train,
        n_estimators=200,
        max_depth=10,
        min_samples_split=4
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    metrics['y_test'] = y_test  # Add y_test to metrics for saving
    
    # Plot feature importance
    plot_feature_importance(model, X_train.columns)
    
    # Save results
    save_model_results(model, metrics)
    logger.info("Model training pipeline completed successfully!")

    logger.info("=================Evaluating the model...=====================")
    y_pred = model.predict(X_test)  # Get predictions for evaluation
    evaluate_model(model, X_test, y_test)
    logger.info("Model evaluation completed successfully!")
    logger.info("===================================================================")  

    logger.info("=================Plotting predictions...=====================")
    plot_predictions(y_test, y_pred)
    logger.info("Predictions plotting completed successfully!")
    logger.info("===================================================================")

    logger.info("=================Plotting residuals...=====================")
    plot_residuals(y_test, y_pred)
    logger.info("Residuals plotting completed successfully!")
    logger.info("===================================================================")  

    logger.info("=================Saving evaluation results...=====================")
    save_evaluation_results(metrics)
    logger.info("Evaluation results saving completed successfully!")
    logger.info("===================================================================")  

    logger.info("=================Main function completed successfully!=====================")
    logger.info("===================================================================")      
    
if __name__ == "__main__":
    main()  