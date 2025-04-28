import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from src.utils import logger

def preprocess_data(dataset):
    # Step 1: Load the dataset and remove unnecessary columns
    logger.info("Step 1: Load the dataset, display information about the dataset before preprocessing and remove unnecessary columns")
    dataset = dataset.drop(columns=["Unnamed: 15", "Unnamed: 16"], errors='ignore')
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())

    # Step 2: Handle missing values in the dataset
    logger.info("Step 2: Handle missing values in the dataset")
    for column in dataset.columns:
        if pd.api.types.is_numeric_dtype(dataset[column]):  
            # Fill missing values in numeric columns with the mean
            dataset[column] = dataset[column].fillna(dataset[column].mean())
        elif pd.api.types.is_object_dtype(dataset[column]):
            # Fill missing values in non-numeric columns with the most frequent value
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])

    # Step 3: Convert object columns to appropriate data types
    logger.info("Step 3: Convert object columns to appropriate data types")
    for column in dataset.columns:
        if dataset[column].dtype == "object":
            if column == "Date":
                # Convert 'Date' column from DD/MM/YYYY format to datetime
                dataset[column] = pd.to_datetime(dataset[column], format='%d/%m/%Y', errors="coerce")
            elif column == "Time":
                # Convert 'Time' column from HH.MM.SS format to time
                dataset[column] = pd.to_datetime(dataset[column], format='%H.%M.%S', errors="coerce").dt.time
            else:
                # Replace commas with dots for decimal numbers and convert to numeric
                dataset[column] = dataset[column].str.replace(',', '.', regex=False)
                dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
                if column == "CO(GT)":
                    dataset[column] = dataset[column].replace(-200, dataset[column].mean())

    # Handle NMHC(GT) column specifically before outlier detection
    if "NMHC(GT)" in dataset.columns:
        # Replace negative values with NaN
        dataset["NMHC(GT)"] = dataset["NMHC(GT)"].mask(dataset["NMHC(GT)"] < 0)
        # Fill NaN values with the mean of the positive values
        positive_mean = dataset[dataset["NMHC(GT)"] > 0]["NMHC(GT)"].mean()
        dataset["NMHC(GT)"] = dataset["NMHC(GT)"].fillna(positive_mean)

    # Step 4: Identify and handle outliers in the dataset
    logger.info("Step 4: Identify and handle outliers in the dataset")
    for column in dataset.select_dtypes(include=[np.number]).columns:
        if column != "NMHC(GT)":  # Skip NMHC(GT) as it's handled separately
            # Calculate Z-scores to identify outliers
            z_scores = np.abs(dataset[column] - dataset[column].mean()) / dataset[column].std()
            # Define a threshold for identifying outliers (3.5 standard deviations)
            threshold = 3.5
            
            # Replace outliers with the median value of the column
            dataset[column] = np.where(z_scores > threshold, dataset[column].median(), dataset[column])
            
            # Optional: Replace negative values (if they are not reasonable for that column) with the median
            if column not in ["NOx(GT)", "NO2(GT)"]:  # These columns can have negative values
                dataset[column] = np.where(dataset[column] < 0, dataset[column].median(), dataset[column])

    # Handle NMHC(GT) outliers separately
    if "NMHC(GT)" in dataset.columns:
        z_scores = np.abs(dataset["NMHC(GT)"] - dataset["NMHC(GT)"].mean()) / dataset["NMHC(GT)"].std()
        threshold = 3.5
        # Replace outliers with the median of positive values
        positive_median = dataset[dataset["NMHC(GT)"] > 0]["NMHC(GT)"].median()
        dataset["NMHC(GT)"] = np.where(z_scores > threshold, positive_median, dataset["NMHC(GT)"])
        # Ensure no negative values
        dataset["NMHC(GT)"] = np.maximum(dataset["NMHC(GT)"], 0)

    # Step 5: Feature engineering - create new features from existing data
    logger.info("Step 5: Feature engineering - create new features from existing data")
    # Combine 'Date' and 'Time' into a single 'Date_Time' column and drop the original columns
    dataset['Date_Time'] = pd.to_datetime(
        dataset['Date'].dt.strftime('%Y-%m-%d') + ' ' +     
        dataset['Time'].astype(str),
        errors='coerce'
    )
    dataset = dataset.drop(columns=['Date', 'Time'], errors='ignore')
    
    # Extract year, month, day, and hour from the 'Date_Time' column
    dataset['Year'] = dataset['Date_Time'].dt.year
    dataset['Month'] = dataset['Date_Time'].dt.month
    dataset['Day'] = dataset['Date_Time'].dt.day
    dataset['Hour'] = dataset['Date_Time'].dt.hour
    
    # Add new interaction features
    dataset['NOx_Temp'] = dataset['NOx(GT)'] * dataset['T']
    dataset['NO2_Humidity'] = dataset['NO2(GT)'] * dataset['RH']
    dataset['NOx_squared'] = dataset['NOx(GT)'] ** 2
    dataset['NO2_squared'] = dataset['NO2(GT)'] ** 2
    
    # Add rolling window features with larger window
    window_size = 5  # Increased from 3 to 5
    for col in ['T', 'RH', 'NOx(GT)', 'NO2(GT)', 'CO(GT)']:
        dataset[f'{col}_rolling_mean'] = dataset[col].rolling(window=window_size).mean()
        dataset[f'{col}_rolling_std'] = dataset[col].rolling(window=window_size).std()
    
    # Fill NaN values created by rolling windows
    dataset = dataset.fillna(method='bfill')

    # Display information about the dataset after preprocessing
    logger.info("Step 6: Display information about the dataset after preprocessing")
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())

    return dataset