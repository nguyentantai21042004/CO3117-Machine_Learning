# Air Quality Analysis Project

This project focuses on analyzing and processing air quality data from the AirQualityUCI dataset. It includes data preprocessing, feature engineering, and model development for air quality prediction.

## 4. Data Preprocessing Pipeline

### 4.1 Overview
The data preprocessing pipeline consists of six main steps, each addressing specific aspects of data quality and feature engineering. The pipeline is implemented in `data_preprocessing.py` and handles various data issues while creating new meaningful features.

### 4.2 Step-by-Step Process

#### Step 1: Initial Data Cleaning
- **Objective**: Remove unnecessary columns and prepare the dataset for further processing
- **Actions**:
  - Remove columns "Unnamed: 15" and "Unnamed: 16" which contain no meaningful data
  - Use `errors='ignore'` to handle cases where these columns don't exist
- **Impact**: Reduces dataset size and removes irrelevant information

#### Step 2: Missing Value Handling
- **Objective**: Address missing values in the dataset systematically
- **Actions**:
  - For numeric columns: Fill missing values with the mean of the column
  - For non-numeric columns: Fill missing values with the most frequent value (mode)
- **Impact**: Ensures data completeness while maintaining statistical properties

#### Step 3: Data Type Conversion
- **Objective**: Convert data types to appropriate formats for analysis
- **Actions**:
  - Date Conversion: Convert 'Date' column from DD/MM/YYYY format to datetime
  - Time Conversion: Convert 'Time' column from HH.MM.SS format to time
  - Numeric Conversion: 
    - Replace commas with dots for decimal numbers
    - Convert string values to numeric
    - Special handling for CO(GT) column: Replace -200 values with column mean
- **Impact**: Ensures consistent data types for analysis and modeling

#### Step 4: Outlier Detection and Treatment
- **Objective**: Identify and handle outliers to improve data quality
- **Actions**:
  - Calculate Z-scores for each numeric column
  - Define threshold of 3 standard deviations for outlier detection
  - Replace outliers with column mean
  - Special handling for NMHC(GT) column:
    - Replace negative values with NaN
    - Fill NaN values with mean of positive values
    - Apply separate outlier treatment
    - Ensure no negative values remain
- **Impact**: Reduces the influence of extreme values on model performance

#### Step 5: Feature Engineering
- **Objective**: Create new meaningful features from existing data
- **Actions**:
  - Temporal Features:
    - Combine 'Date' and 'Time' into 'Date_Time' column
    - Extract year, month, day, and hour components
  - Drop original 'Date' and 'Time' columns
- **Impact**: Creates more informative features for time-series analysis

#### Step 6: Data Validation and Summary
- **Objective**: Verify preprocessing results and provide dataset summary
- **Actions**:
  - Display first few rows of processed dataset
  - Show dataset information (data types, non-null counts)
  - Generate descriptive statistics
- **Impact**: Ensures preprocessing quality and provides insights into the final dataset

### 4.3 Key Considerations

#### Data Quality
- Special attention given to NMHC(GT) column due to its unique characteristics
- Systematic approach to handling missing values and outliers
- Preservation of data distribution through mean-based imputation

#### Feature Engineering
- Creation of temporal features for time-series analysis
- Proper handling of date and time formats
- Removal of redundant columns

#### Validation
- Comprehensive logging of each preprocessing step
- Multiple validation points through dataset summaries
- Clear documentation of transformations applied

### 4.4 Technical Implementation Details

#### Libraries Used
- pandas: For data manipulation and analysis
- numpy: For numerical operations
- Custom logging module: For tracking preprocessing steps

#### Error Handling
- Graceful handling of missing columns
- Robust type conversion with error handling
- Systematic approach to outlier treatment

#### Performance Considerations
- Efficient handling of large datasets
- Vectorized operations for better performance
- Memory-efficient transformations

### 4.5 Output
The preprocessing pipeline produces a clean, well-structured dataset ready for:
- Feature selection
- Model training
- Time-series analysis
- Statistical analysis

The processed dataset maintains the original data's integrity while improving its quality and adding valuable features for analysis.

## 5. Data Splitting and Feature Engineering Pipeline

### 5.1 Overview
The data splitting pipeline is implemented in `data_splits.py` and serves as the second step in our data processing workflow. It focuses on advanced feature engineering and proper data partitioning for model training and evaluation.

### 5.2 Feature Engineering Process

#### 5.2.1 Interaction Features
- **Objective**: Capture relationships between related variables
- **Created Features**:
  - `Temp_Humidity`: Interaction between temperature and relative humidity
  - `NOx_NO2`: Interaction between NOx and NO2 concentrations
- **Impact**: Captures non-linear relationships between environmental factors

#### 5.2.2 Polynomial Features
- **Objective**: Model non-linear relationships in the data
- **Created Features**:
  - `Temp_squared`: Quadratic term for temperature
  - `RH_squared`: Quadratic term for relative humidity
- **Impact**: Enables modeling of quadratic relationships in environmental variables

#### 5.2.3 Time-Based Features
- **Objective**: Capture cyclical patterns in time-based data
- **Created Features**:
  - `Hour_sin` and `Hour_cos`: Cyclical encoding of hour (24-hour cycle)
  - `Month_sin` and `Month_cos`: Cyclical encoding of month (12-month cycle)
- **Impact**: Preserves temporal relationships while enabling linear models to capture cyclical patterns

#### 5.2.4 Ratio Features
- **Objective**: Create normalized relationships between variables
- **Created Features**:
  - `NOx_NO2_ratio`: Ratio between NOx and NO2 concentrations
  - `Temp_RH_ratio`: Ratio between temperature and relative humidity
- **Impact**: Provides normalized measures of relationships between environmental factors

#### 5.2.5 Rolling Window Features
- **Objective**: Capture temporal dependencies and trends
- **Created Features**:
  - Rolling mean and standard deviation for:
    - Temperature
    - Relative Humidity
    - NOx concentration
    - NO2 concentration
- **Window Size**: 3 samples
- **Impact**: Captures short-term trends and variations in environmental measurements

### 5.3 Data Splitting Process

#### 5.3.1 Feature and Target Preparation
- **Feature Selection**:
  - Select all numeric columns except the target variable
  - Exclude 'CO(GT)' from features
  - Ensure only numeric data types are included
- **Target Variable**:
  - Use the last column as the target variable
  - Maintain original scale for target values

#### 5.3.2 Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)
- **Stratification**: Not applied (continuous target variable)
- **Validation**: 
  - Logging of split sizes
  - Feature count verification
  - Data type consistency checks

#### 5.3.3 Feature Scaling
- **Method**: StandardScaler
- **Process**:
  - Fit scaler on training data only
  - Transform both training and testing data
  - Maintain column names through DataFrame conversion
- **Impact**: 
  - Normalizes feature scales
  - Improves model convergence
  - Prevents feature dominance

### 5.4 Technical Implementation Details

#### 5.4.1 Libraries Used
- pandas: For data manipulation
- numpy: For numerical operations
- sklearn: For data splitting and scaling
- Custom logging module: For process tracking

#### 5.4.2 Error Handling
- Comprehensive try-except blocks
- Detailed error logging
- Graceful failure handling

#### 5.4.3 Performance Considerations
- Efficient DataFrame operations
- Vectorized calculations
- Memory-efficient transformations

### 5.5 Output and Usage

#### 5.5.1 Return Values
The pipeline returns:
- `X_train_scaled`: Scaled training features
- `X_test_scaled`: Scaled testing features
- `y_train`: Training target values
- `y_test`: Testing target values
- `scaler`: Fitted StandardScaler object

#### 5.5.2 Usage in Model Training
- Direct integration with model training pipeline
- Consistent feature scaling across training and testing
- Preservation of feature names for interpretability

### 5.6 Key Considerations

#### 5.6.1 Data Quality
- Handling of NaN values in rolling features
- Proper scaling of all features
- Maintenance of data integrity

#### 5.6.2 Feature Engineering
- Creation of meaningful interactions
- Proper handling of cyclical features
- Appropriate window size for rolling features

#### 5.6.3 Validation
- Comprehensive logging at each step
- Verification of split sizes
- Consistency checks for data types

### 5.7 Impact on Model Performance
The data splitting and feature engineering pipeline significantly impacts model performance by:
- Creating informative features for prediction
- Ensuring proper data partitioning
- Maintaining data quality and consistency
- Providing normalized features for model training

## 6. Model Training and Evaluation Pipeline

### 6.1 Overview
The model training pipeline is implemented in `train.py` and focuses on training a Random Forest model for air quality prediction. The pipeline includes model training, evaluation, visualization, and result saving components.

### 6.2 Model Training Process

#### 6.2.1 Model Initialization
- **Algorithm**: Random Forest Regressor
- **Key Parameters**:
  - `n_estimators`: Number of trees in the forest (default: 100)
  - `max_depth`: Maximum depth of the tree (default: None)
  - `min_samples_split`: Minimum samples required to split a node (default: 2)
  - `random_state`: Seed for reproducibility (default: 42)
- **Implementation**:
  - Comprehensive error handling
  - Detailed logging of parameters
  - Progress tracking during training

#### 6.2.2 Training Process
- **Input Data**:
  - `X_train`: Scaled training features
  - `y_train`: Training target values
- **Process**:
  - Model initialization with specified parameters
  - Training on the prepared dataset
  - Progress logging at each stage
- **Output**:
  - Trained Random Forest model
  - Training completion confirmation

### 6.3 Model Evaluation

#### 6.3.1 Performance Metrics
- **Metrics Calculated**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- **Implementation**:
  - Prediction on test set
  - Metric calculation
  - Detailed logging of results
- **Output**:
  - Dictionary containing all metrics
  - Predicted values for visualization

#### 6.3.2 Feature Importance Analysis
- **Visualization**:
  - Horizontal bar plot of top N features
  - Sorted by importance
  - Clear labeling and formatting
- **Implementation**:
  - Extraction of feature importances
  - Sorting and selection of top features
  - Professional plotting with matplotlib
- **Output**:
  - Saved plot in PNG format
  - Logging of saved file location

### 6.4 Results Management

#### 6.4.1 Result Saving
- **Saved Components**:
  - Model metrics (CSV format)
  - Actual vs. Predicted values (CSV format)
  - Feature importance plot (PNG format)
- **Directory Management**:
  - Automatic creation of results directory
  - Organized file structure
  - Clear naming conventions

#### 6.4.2 Error Handling
- **Comprehensive Error Management**:
  - Try-except blocks for all operations
  - Detailed error logging
  - Graceful failure handling
- **Validation**:
  - Directory existence checks
  - File writing verification
  - Process completion confirmation

### 6.5 Technical Implementation Details

#### 6.5.1 Libraries Used
- pandas: For data handling and CSV operations
- numpy: For numerical computations
- matplotlib: For visualization
- scikit-learn: For model implementation and metrics
- Custom logging module: For process tracking

#### 6.5.2 Performance Considerations
- Efficient memory usage
- Vectorized operations
- Proper resource cleanup (plot closing)

#### 6.5.3 Code Organization
- Modular function design
- Clear documentation
- Consistent error handling
- Comprehensive logging

### 6.6 Output and Usage

#### 6.6.1 Generated Files
- `model_metrics.csv`: Contains evaluation metrics
- `predictions.csv`: Contains actual and predicted values
- `feature_importance.png`: Visualization of feature importance

#### 6.6.2 Directory Structure
```
assignment/
└── train_results/
    ├── model_metrics.csv
    ├── predictions.csv
    └── feature_importance.png
```

### 6.7 Key Considerations

#### 6.7.1 Model Selection
- Random Forest chosen for:
  - Handling non-linear relationships
  - Feature importance analysis
  - Robustness to outliers

#### 6.7.2 Evaluation Strategy
- Comprehensive metric suite
- Visual analysis of predictions
- Feature importance interpretation

#### 6.7.3 Result Management
- Organized file structure
- Clear documentation
- Easy access to results

### 6.8 Impact on Project

The model training pipeline provides:
- Reliable model training process
- Comprehensive evaluation metrics
- Visual insights into model behavior
- Organized result management
- Foundation for model improvement

## 7. Model Evaluation Pipeline

### 7.1 Overview
The model evaluation pipeline is implemented in `evaluate.py` and provides a comprehensive framework for assessing model performance. It includes metric calculation, visualization, and result management components to thoroughly analyze the model's predictive capabilities.

### 7.2 Performance Metrics Analysis

#### 7.2.1 Metric Calculation
- **Metrics Implemented**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
- **Implementation Details**:
  - Precise calculation using scikit-learn metrics
  - High precision logging (6 decimal places)
  - Comprehensive error handling
- **Output Format**:
  - Dictionary containing all metrics
  - Detailed logging of each metric
  - Structured for easy analysis

#### 7.2.2 Metric Interpretation
- **MSE/RMSE**:
  - Measures average squared difference
  - RMSE provides interpretable units
  - Sensitive to outliers
- **MAE**:
  - Average absolute difference
  - More robust to outliers
  - Directly interpretable
- **R²**:
  - Proportion of variance explained
  - Scale-independent
  - Range: 0 to 1

### 7.3 Visualization Components

#### 7.3.1 Predictions Comparison Plot
- **Visualization Type**: Line plot
- **Features**:
  - Actual vs. Predicted values
  - Color-coded for clarity
  - Grid for better readability
- **Technical Details**:
  - Figure size: 12x6 inches
  - Alpha value: 0.7 for transparency
  - Professional labeling
- **Output**:
  - PNG format
  - High resolution
  - Clear legend

#### 7.3.2 Residuals Analysis Plot
- **Visualization Type**: Scatter plot
- **Features**:
  - Residuals vs. Actual values
  - Zero reference line
  - Grid for better analysis
- **Technical Details**:
  - Figure size: 12x6 inches
  - Alpha value: 0.5 for density
  - Professional formatting
- **Output**:
  - PNG format
  - High resolution
  - Clear annotations

### 7.4 Results Management

#### 7.4.1 File Organization
- **Directory Structure**:
  ```
  assignment/
  └── evaluate_results/
      ├── evaluation_metrics.csv
      ├── predictions_comparison.png
      └── residuals_plot.png
  ```
- **File Formats**:
  - CSV for metrics
  - PNG for visualizations
- **Naming Conventions**:
  - Clear, descriptive names
  - Consistent formatting
  - Easy to locate

#### 7.4.2 Data Storage
- **Metrics Storage**:
  - Structured CSV format
  - Single row per evaluation
  - All metrics included
- **Visualization Storage**:
  - High-quality PNG files
  - Proper aspect ratios
  - Clear labeling

### 7.5 Technical Implementation

#### 7.5.1 Libraries Used
- pandas: For data handling
- numpy: For numerical operations
- matplotlib: For visualization
- scikit-learn: For metric calculation
- Custom logging module: For tracking

#### 7.5.2 Error Handling
- **Comprehensive Error Management**:
  - Try-except blocks
  - Detailed error messages
  - Graceful failure handling
- **Validation**:
  - Input data checks
  - Directory existence verification
  - File writing confirmation

#### 7.5.3 Performance Optimization
- **Memory Management**:
  - Proper plot closing
  - Efficient data structures
  - Resource cleanup
- **Processing Efficiency**:
  - Vectorized operations
  - Optimized calculations
  - Minimal memory usage

### 7.6 Pipeline Execution

#### 7.6.1 Data Loading
- **Input Sources**:
  - Test data from split_data directory
  - Predictions from train_results
- **Data Validation**:
  - Shape verification
  - Type checking
  - Missing value handling

#### 7.6.2 Process Flow
1. Load test data and predictions
2. Calculate performance metrics
3. Generate visualizations
4. Save results
5. Log completion

### 7.7 Key Considerations

#### 7.7.1 Metric Selection
- Comprehensive coverage of error types
- Balance between sensitivity and robustness
- Clear interpretation guidelines

#### 7.7.2 Visualization Design
- Professional appearance
- Clear communication
- Appropriate scales
- Informative labels

#### 7.7.3 Result Management
- Organized storage
- Easy access
- Clear documentation
- Proper versioning

### 7.8 Impact on Project

The evaluation pipeline provides:
- Thorough model assessment
- Clear performance visualization
- Organized result management
- Foundation for model improvement
- Documentation of model capabilities
