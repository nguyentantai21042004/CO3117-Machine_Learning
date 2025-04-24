import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from src.utils import logger

def feature_engineering(df):
    """
    Perform advanced feature engineering to create more informative features.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with new features
    """
    try:
        logger.info("Performing advanced feature engineering...")
        
        # Create interaction features
        df['Temp_Humidity'] = df['T'] * df['RH']  # T is Temperature, RH is Relative Humidity
        df['NOx_NO2'] = df['NOx(GT)'] * df['NO2(GT)']
        
        # Create polynomial features for important variables
        df['Temp_squared'] = df['T'] ** 2
        df['RH_squared'] = df['RH'] ** 2
        
        # Create time-based features
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        
        # Create ratio features
        df['NOx_NO2_ratio'] = df['NOx(GT)'] / (df['NO2(GT)'] + 1e-6)  # Add small constant to avoid division by zero
        df['Temp_RH_ratio'] = df['T'] / (df['RH'] + 1e-6)
        
        # Create rolling window features
        window_size = 3
        for col in ['T', 'RH', 'NOx(GT)', 'NO2(GT)']:  # Updated column names
            df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
        
        # Fill NaN values created by rolling windows
        df = df.fillna(method='bfill')
        
        logger.info("Feature engineering completed successfully!")
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def split_data(dataset):
    """
    Step 2: Split the preprocessed dataset into training and testing sets.
    This function assumes the data has already been preprocessed in data_preprocessing.py.
    
    Steps:
    1. Perform feature engineering
    2. Split data into train and test sets
    3. Scale the features using StandardScaler
    
    Args:
        dataset (pd.DataFrame): A DataFrame containing the preprocessed features and a Series for the target variable.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    try:
        logger.info("Starting Step 2: Data Splitting Process...")

        # Perform feature engineering
        dataset = feature_engineering(dataset)

        # Unpack the dataset
        X = dataset.iloc[:, :-1]  # All columns except the last one as features
        y = dataset.iloc[:, -1]    # The last column as the target variable
        
        # Prepare features and target
        logger.info("Preparing features and target variable...")    
            
        # Select only numeric columns for features, excluding CO(GT)
        numeric_columns = X.select_dtypes(include=['float64', 'int32']).columns
        numeric_columns = numeric_columns[numeric_columns != 'CO(GT)']  # Exclude CO(GT)
        
        X = X[numeric_columns]
        
        # Log feature information
        logger.info(f"Number of features: {X.shape[1]}")
        logger.info(f"Feature names: {list(X.columns)}")
        
        # Split the data
        logger.info("Splitting data into train (80%) and test (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Log split information
        logger.info(f"Training set size: {X_train.shape[0]} samples")
        logger.info(f"Test set size: {X_test.shape[0]} samples")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        # Scale the features
        logger.info("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        logger.info("Data splitting completed successfully!")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Error during data splitting: {str(e)}")
        raise