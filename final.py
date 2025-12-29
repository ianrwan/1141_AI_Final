import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# 1. Data Loading
def load_wine_data(file_path='dataset/winequality-red.csv'):
    """
    Load wine quality dataset
    Note: Data uses semicolon as delimiter
    """
    try:
        # Since data uses semicolon delimiter, we specify separator
        df = pd.read_csv(file_path, sep=';')
        print(f"Data loaded successfully! Total {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("File not found, please check the file path")
        return None

# 2. Data Exploration
def explore_data(df):
    """Exploratory Data Analysis"""
    print("\n=== Data Exploration ===")
    
    # Display basic information
    print("1. First 5 rows of data:")
    print(df.head())
    
    print("\n2. Basic data information:")
    print(f"Data shape: {df.shape}")
    
    print("\n3. Data types:")
    print(df.dtypes)
    
    print("\n4. Statistical summary:")
    print(df.describe())
    
    # Check for missing values
    print("\n5. Missing values check:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values in data!")
    else:
        print("Missing values situation:")
        print(missing_values[missing_values > 0])
    
    return df

# 3. Data Cleaning
def clean_data(df):
    """
    Clean data
    Based on textbook principles:
    1. Handle duplicate values
    2. Handle outliers
    """
    print("\n=== Data Cleaning ===")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows")
    
    if duplicates > 0:
        df_cleaned = df.drop_duplicates()
        print(f"After removing duplicates, data shape: {df_cleaned.shape}")
    else:
        df_cleaned = df.copy()
    
    # Detect outliers (using IQR method)
    print("\nOutlier detection (using IQR method):")
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = ((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).sum()
    print("Number of outliers per feature:")
    print(outliers)
    
    # Optional: Remove extreme outliers
    # Here we choose to keep outliers as in wine data, extreme values might represent special quality wines
    
    return df_cleaned

# 4. Feature Engineering
def feature_engineering(df):
    """
    Feature Engineering
    Create new features based on wine chemistry knowledge
    """
    print("\n=== Feature Engineering ===")
    
    df_engineered = df.copy()
    
    # 1. Create total acidity feature
    df_engineered['total_acidity'] = (
        df_engineered['fixed acidity'] + 
        df_engineered['volatile acidity'] + 
        df_engineered['citric acid']
    )
    print("Created new feature: Total Acidity (total_acidity)")
    
    # 2. Create SO2 ratio feature
    df_engineered['free_SO2_ratio'] = (
        df_engineered['free sulfur dioxide'] / 
        df_engineered['total sulfur dioxide']
    )
    # Handle division by zero errors
    df_engineered['free_SO2_ratio'] = df_engineered['free_SO2_ratio'].replace([np.inf, -np.inf], np.nan)
    df_engineered['free_SO2_ratio'] = df_engineered['free_SO2_ratio'].fillna(0)
    print("Created new feature: Free SO2 Ratio (free_SO2_ratio)")
    
    # 3. Create acid-base balance feature
    df_engineered['acid_balance'] = (
        df_engineered['fixed acidity'] / 
        (df_engineered['pH'] + 1e-10)  # Avoid division by zero
    )
    print("Created new feature: Acid Balance (acid_balance)")
    
    # 4. Create alcohol to acidity ratio
    df_engineered['alcohol_acidity_ratio'] = (
        df_engineered['alcohol'] / 
        (df_engineered['total_acidity'] + 1e-10)
    )
    print("Created new feature: Alcohol Acidity Ratio (alcohol_acidity_ratio)")
    
    print(f"After feature engineering, data shape: {df_engineered.shape}")
    print(f"Number of new features: {len(df_engineered.columns) - len(df.columns)}")
    
    return df_engineered

# 5. Feature Selection
def select_features(df, target_column='quality', name='red'):
    """
    Feature Selection
    Use correlation analysis and domain knowledge to select important features
    """
    print("\n=== Feature Selection ===")
    
    # Calculate correlation with target variable
    correlation = df.corr()[target_column].abs().sort_values(ascending=False)
    
    print("Feature correlation with target variable (quality):")
    print(correlation)
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Select features based on correlation (threshold can be adjusted)
    # Here we select features with correlation greater than 0.05
    selected_features = correlation[correlation > 0.1].index.tolist()
    selected_features.remove(target_column)  # Remove target variable
    
    print(f"\nSelected {len(selected_features)} features:")
    print(selected_features)
    
    return selected_features

# 6. Data Standardization/Normalization
def scale_features(df, features, target_column='quality'):
    print("Standardization Features:\n", features)
    """
    Feature Scaling
    Use standardization (Z-score normalization) to make features have zero mean and unit variance
    """
    print("\n=== Feature Scaling ===")
    
    # Separate features and target
    X = df[features]
    print("Standardization X:\n",X)
    y = df[target_column]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform features
    X_scaled = scaler.fit_transform(X)
    print("Standardization X_scaled:\n",X_scaled)
    
    # Create new DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    print("Standardization X_scaled_df:\n",X_scaled_df)
    
    print("Scaling completed!")
    print("Statistical summary after scaling:")
    print(pd.DataFrame(X_scaled_df).describe().loc[['mean', 'std']])
    
    return X_scaled_df, y, scaler

# 7. Handle Categorical Variables (if needed)
def encode_categorical(df, columns=None):
    """
    Encode categorical variables
    """
    print("\n=== Categorical Variable Encoding ===")
    
    df_encoded = df.copy()
    
    if columns:
        for col in columns:
            if df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                print(f"Encoded column: {col}")
    
    return df_encoded

# 8. Data Splitting
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    print("\n=== Data Splitting ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Training set target distribution: \n{y_train.value_counts().sort_index()}")
    print(f"Testing set target distribution: \n{y_test.value_counts().sort_index()}")
    
    return X_train, X_test, y_train, y_test

# 9. Principal Component Analysis (Optional, for visualization)
def apply_pca(X, n_components=2):
    """
    Apply PCA for dimensionality reduction and visualization
    """
    print("\n=== PCA Dimensionality Reduction ===")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, pca

# 10. Complete Preprocessing Pipeline
def complete_preprocessing_pipeline(file_path='dataset/winequality-red.csv', test_size=0.2, name='red'):
    """
    Complete data preprocessing pipeline
    """
    print("="*50)
    print("Starting Data Preprocessing Pipeline")
    print("="*50)
    
    # Step 1: Load data
    df = load_wine_data(file_path)
    if df is None:
        return None    
    
    # Step 2: Data exploration
    df = explore_data(df)
    
    # Step 3: Data cleaning
    df_cleaned = clean_data(df)
    
    # Step 4: Feature engineering
    df_engineered = feature_engineering(df_cleaned)
    
    # Step 5: Handle categorical variables (this dataset has none, so skip)
    
    # Step 6: Feature selection
    selected_features = select_features(df_engineered, name=name)
    
    # Step 7: Feature scaling
    X_scaled, y, scaler = scale_features(df_engineered, selected_features)
    
    # Step 8: Data splitting
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size)
    
    # Step 9: Optional PCA analysis
    X_pca, pca = apply_pca(X_scaled)
    
    print("\n" + "="*50)
    print("Preprocessing Pipeline Completed!")
    print("="*50)
    
    # Return all necessary results
    results = {
        'original_data': df,
        'cleaned_data': df_cleaned,
        'engineered_data': df_engineered,
        'selected_features': selected_features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'pca': pca,
        'X_pca': X_pca
    }
    
    return results

# 11. Visualize Preprocessing Results
def visualize_preprocessing_results(results, name='red'):
    """
    Visualize key results from preprocessing
    """
    print("\n=== Preprocessing Results Visualization ===")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Quality distribution
    axes[0, 0].hist(results['original_data']['quality'], bins=range(3, 10), edgecolor='black')
    axes[0, 0].set_title('Wine Quality Distribution')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Alcohol content distribution
    axes[0, 1].hist(results['original_data']['alcohol'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Alcohol Content Distribution')
    axes[0, 1].set_xlabel('Alcohol Content')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Acidity distribution
    axes[0, 2].hist(results['original_data']['fixed acidity'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Fixed Acidity Distribution')
    axes[0, 2].set_xlabel('Fixed Acidity')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Relationship between quality and alcohol
    axes[1, 0].scatter(results['original_data']['quality'], 
                      results['original_data']['alcohol'], 
                      alpha=0.5)
    axes[1, 0].set_title('Quality vs Alcohol Content')
    axes[1, 0].set_xlabel('Quality Score')
    axes[1, 0].set_ylabel('Alcohol Content')
    
    # 5. Relationship between quality and acidity
    axes[1, 1].scatter(results['original_data']['quality'], 
                      results['original_data']['fixed acidity'], 
                      alpha=0.5)
    axes[1, 1].set_title('Quality vs Fixed Acidity')
    axes[1, 1].set_xlabel('Quality Score')
    axes[1, 1].set_ylabel('Fixed Acidity')
    
    # 6. PCA visualization (if available)
    if 'X_pca' in results and results['X_pca'] is not None:
        # Get the quality values from the engineered data (which has the same length as X_pca)
        # The engineered_data is already cleaned (duplicates removed)
        quality_for_pca = results['engineered_data']['quality']
        
        # Ensure the lengths match
        if len(quality_for_pca) == len(results['X_pca']):
            scatter = axes[1, 2].scatter(results['X_pca'][:, 0], 
                                        results['X_pca'][:, 1], 
                                        c=quality_for_pca, 
                                        cmap='viridis', 
                                        alpha=0.7)
            axes[1, 2].set_title('PCA Dimensionality Reduction Visualization')
            axes[1, 2].set_xlabel('Principal Component 1')
            axes[1, 2].set_ylabel('Principal Component 2')
            plt.colorbar(scatter, ax=axes[1, 2], label='Quality Score')
        else:
            print(f"Warning: Length mismatch for PCA plot. X_pca has {len(results['X_pca'])} samples, "
                  f"but quality has {len(quality_for_pca)} samples.")
            axes[1, 2].text(0.5, 0.5, 'PCA Visualization\n(Data length mismatch)', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('PCA Dimensionality Reduction')
    else:
        axes[1, 2].text(0.5, 0.5, 'PCA Not Available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('PCA Dimensionality Reduction')
    
    plt.tight_layout()
    plt.savefig(f'chart_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def result_output(results, name='red'):
    if results:
        # Visualize results
        visualize_preprocessing_results(results, name)
        
        # Output summary of preprocessed data
        print("\nPreprocessing Results Summary:")
        print(f"Original data shape: {results['original_data'].shape}")
        print(f"Cleaned data shape: {results['cleaned_data'].shape}")
        print(f"Engineered data shape: {results['engineered_data'].shape}")
        print(f"Number of selected features: {len(results['selected_features'])}")
        print(f"Training set shape: {results['X_train'].shape}")
        print(f"Testing set shape: {results['X_test'].shape}")

        # Concat X and y Data
        train_result = results['X_train'].reset_index(drop=True)
        train_result['quality'] = results['y_train'].reset_index(drop=True)
        print("Show train_result concat X and y Data:\n", train_result)

        test_result = results['X_test'].reset_index(drop=True)
        test_result['quality'] = results['y_test'].reset_index(drop=True)
        print("Show test_result concat X and y Data:\n", test_result)

        # Save preprocessed data
        train_result.to_csv(f'training_data_{name}.csv', index=False)
        test_result.to_csv(f'testing_data_{name}.csv', index=False)
        print("\nPreprocessed data saved as CSV files")

# Main program execution
if __name__ == "__main__":
    # Execute complete preprocessing pipeline
    df_red = load_wine_data('dataset/winequality-red.csv')
    df_white = load_wine_data('dataset/winequality-white.csv')
    df_total = pd.concat([df_red, df_white], ignore_index=True)
    df_total.to_csv(f'winequality-total.csv', sep=';',index=False)
    print(df_red.shape)
    print(df_white.shape)
    print(df_total.shape)
    results_total = complete_preprocessing_pipeline('winequality-total.csv', name='total')
    result_output(results_total, name='total')

    os.remove('winequality-total.csv')
    print('winequality-total.csv is deleted')
    
    

