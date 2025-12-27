# Wine Quality Prediction with XGBoost - Baseline Model (No Preprocessing)
# This script trains an XGBoost model on raw, unprocessed data for comparison with preprocessed results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1. Load Raw Wine Data
def load_raw_wine_data(file_path='dataset/winequality-red.csv'):
    """Load raw wine quality dataset"""
    print("Loading raw wine data...")
    
    try:
        # Note: Data uses semicolon delimiter
        df = pd.read_csv(file_path, sep=';')
        print(f"Raw data loaded successfully! Total {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic information
        print("\n=== Raw Data Information ===")
        print(f"Data shape: {df.shape}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        return df
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None

# 2. Train Baseline XGBoost Model (No Preprocessing)
def train_baseline_xgboost_model(df, test_size=0.2, random_state=42):
    """Train and evaluate XGBoost model on raw data"""
    print("\n" + "="*60)
    print("Training Baseline XGBoost Model (No Preprocessing)")
    print("="*60)
    
    # Prepare data
    # Separate features and target
    # Assuming 'quality' is the target column
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features used: {list(X.columns)}")
    
    # Split data (without stratification to match raw data distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Training target distribution: \n{y_train.value_counts().sort_index()}")
    print(f"Testing target distribution: \n{y_test.value_counts().sort_index()}")
    
    # Initialize XGBoost regressor
    print("\nInitializing XGBoost regression model...")
    
    # Use same parameters as preprocessed model for fair comparison
    try:
        # Try new version approach
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )
        
        print("Training with eval_set monitoring...")
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=50
        )
        
    except TypeError:
        # Fall back to old version approach
        print("Using old XGBoost approach...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Round predictions to nearest integer
    y_test_pred_rounded = np.round(y_test_pred).astype(int)
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, y_test_pred_rounded

# 3. Evaluate Baseline Model Performance
def evaluate_baseline_model(y_true, y_pred, dataset_name="Dataset"):
    """Evaluate regression model performance"""
    print(f"\n=== {dataset_name} Evaluation ===")
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Calculate exact accuracy (rounded predictions)
    y_pred_rounded = np.round(y_pred).astype(int)
    accuracy_exact = np.mean(y_true == y_pred_rounded)
    
    # Calculate prediction within ±1 range
    diff = np.abs(y_true - y_pred_rounded)
    accuracy_within_one = np.mean(diff <= 1)
    
    print(f"Exact prediction accuracy: {accuracy_exact:.2%}")
    print(f"Prediction within ±1 score: {accuracy_within_one:.2%}")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'accuracy_exact': accuracy_exact,
        'accuracy_within_one': accuracy_within_one
    }

# 4. Visualize Baseline Model Results
def visualize_baseline_results(y_test, y_test_pred, y_test_pred_rounded, model, X_train):
    """Visualize baseline model results"""
    print("\n=== Visualizing Baseline Model Results ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Quality')
    axes[0, 0].set_ylabel('Predicted Quality')
    axes[0, 0].set_title('Actual vs Predicted (Baseline)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error Distribution
    residuals = y_test - y_test_pred
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution (Baseline)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    axes[0, 2].barh(range(len(importance_df)), importance_df['importance'])
    axes[0, 2].set_yticks(range(len(importance_df)))
    axes[0, 2].set_yticklabels(importance_df['feature'])
    axes[0, 2].set_xlabel('Feature Importance')
    axes[0, 2].set_title('Feature Importance (Baseline)')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    # 4. Confusion Matrix (Rounded)
    all_classes = np.arange(min(y_test.min(), y_test_pred_rounded.min()), 
                           max(y_test.max(), y_test_pred_rounded.max()) + 1)
    
    cm = confusion_matrix(y_test, y_test_pred_rounded, labels=all_classes)
    
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix (Baseline)')
    axes[1, 0].set_xlabel('Predicted Quality')
    axes[1, 0].set_ylabel('Actual Quality')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                          horizontalalignment="center",
                          color="white" if cm[i, j] > thresh else "black")
    
    axes[1, 0].set_xticks(range(len(all_classes)))
    axes[1, 0].set_yticks(range(len(all_classes)))
    axes[1, 0].set_xticklabels(all_classes)
    axes[1, 0].set_yticklabels(all_classes)
    
    # 5. Accuracy by Quality Score
    quality_scores = np.unique(y_test)
    accuracy_by_quality = []
    
    for score in quality_scores:
        mask = y_test == score
        if mask.any():
            accuracy = np.mean(y_test_pred_rounded[mask] == score)
            accuracy_by_quality.append(accuracy)
    
    axes[1, 1].bar(quality_scores, accuracy_by_quality, alpha=0.7)
    axes[1, 1].set_xlabel('Quality Score')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy by Quality (Baseline)')
    axes[1, 1].set_xticks(quality_scores)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Learning Curve or Feature Summary
    try:
        results = model.evals_result()
        if results:
            train_rmse = results['validation_0']['rmse']
            test_rmse = results['validation_1']['rmse']
            
            epochs = len(train_rmse)
            axes[1, 2].plot(range(epochs), train_rmse, label='Training RMSE')
            axes[1, 2].plot(range(epochs), test_rmse, label='Testing RMSE')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('RMSE')
            axes[1, 2].set_title('Learning Curve (Baseline)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    except:
        axes[1, 2].axis('off')
        top_features = importance_df.head(8)
        feature_text = "Top 8 Features (Baseline):\n\n"
        for _, row in top_features.iterrows():
            feature_text += f"{row.feature}: {row.importance:.4f}\n"
        
        axes[1, 2].text(0.5, 0.5, feature_text,
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=10,
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Feature Importance (Top 8)')
    
    plt.tight_layout()
    plt.suptitle('Baseline Model Results (No Preprocessing)', fontsize=16, y=1.02)
    plt.show()
    
    return importance_df

# 5. Analyze Raw Data Statistics
def analyze_raw_data(df):
    """Analyze statistics of raw data"""
    print("\n" + "="*60)
    print("Raw Data Analysis")
    print("="*60)
    
    # Basic statistics
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing Values:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    # Target distribution
    print(f"\nTarget Variable (Quality) Distribution:")
    print(df['quality'].value_counts().sort_index())
    
    # Feature correlations with target
    print(f"\nCorrelation with Quality (Target):")
    correlations = df.corr()['quality'].abs().sort_values(ascending=False)
    print(correlations)
    
    return correlations

# 6. Main Function for Baseline Model
def main_baseline():
    """Main function for baseline model training"""
    print("="*80)
    print("WINE QUALITY PREDICTION - BASELINE MODEL (NO PREPROCESSING)")
    print("="*80)
    
    # Load raw data
    df = load_raw_wine_data('dataset/winequality-red.csv')
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Analyze raw data
    correlations = analyze_raw_data(df)
    
    # Train baseline model
    (model, X_train, X_test, y_train, y_test, 
     y_train_pred, y_test_pred, y_test_pred_rounded) = train_baseline_xgboost_model(df)
    
    # Evaluate model
    print("\n" + "="*60)
    print("BASELINE MODEL PERFORMANCE")
    print("="*60)
    
    print("Training Set Evaluation:")
    train_metrics = evaluate_baseline_model(y_train, y_train_pred, "Training Set")
    
    print("\nTesting Set Evaluation:")
    test_metrics = evaluate_baseline_model(y_test, y_test_pred, "Testing Set")
    
    # Show sample predictions
    print("\nTesting Set - Sample Predictions (First 10):")
    results_df = pd.DataFrame({
        'Actual': y_test[:10],
        'Predicted': np.round(y_test_pred[:10], 2),
        'Rounded': y_test_pred_rounded[:10],
        'Error': y_test[:10] - y_test_pred_rounded[:10]
    })
    print(results_df)
    
    # Visualize results
    importance_df = visualize_baseline_results(y_test, y_test_pred, y_test_pred_rounded, model, X_train)
    
    # Summary
    print("\n" + "="*80)
    print("BASELINE MODEL SUMMARY")
    print("="*80)
    
    print(f"Dataset: winequality-red.csv")
    print(f"Total Samples: {df.shape[0]}")
    print(f"Features Used: {X_train.shape[1]} (all original features)")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    print(f"\nPerformance Metrics (Testing Set):")
    print(f"  - Exact Accuracy: {test_metrics['accuracy_exact']:.2%}")
    print(f"  - Accuracy within ±1: {test_metrics['accuracy_within_one']:.2%}")
    print(f"  - MAE: {test_metrics['mae']:.4f}")
    print(f"  - RMSE: {test_metrics['rmse']:.4f}")
    print(f"  - R² Score: {test_metrics['r2']:.4f}")
    
    print(f"\nTop 5 Most Important Features (Baseline):")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
    
    # Save baseline results for comparison
    baseline_results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'importance_df': importance_df,
        'correlations': correlations
    }
    
    # Save predictions to CSV for comparison
    predictions_df = pd.DataFrame({
        'actual_quality': y_test,
        'predicted_quality': y_test_pred,
        'predicted_quality_rounded': y_test_pred_rounded
    })
    predictions_df.to_csv('baseline_predictions.csv', index=False)
    print(f"\nBaseline predictions saved to 'baseline_predictions.csv'")
    
    print("\n" + "="*80)
    print("COMPARISON NOTES:")
    print("="*80)
    print("1. This baseline model uses RAW DATA without any preprocessing")
    print("2. No feature engineering, selection, or scaling was performed")
    print("3. All original features were used")
    print("4. The model uses the same XGBoost parameters as the preprocessed model")
    print("5. Compare these results with the preprocessed model to see the impact of preprocessing")
    print("\nKey metrics to compare:")
    print("  - Exact accuracy (higher is better)")
    print("  - Accuracy within ±1 score (higher is better)")
    print("  - RMSE (lower is better)")
    print("  - R² score (closer to 1 is better)")
    
    return baseline_results

# 7. Quick Comparison Function (to be used after both models are trained)
def quick_comparison(baseline_metrics, preprocessed_metrics=None):
    """Quick comparison between baseline and preprocessed models"""
    print("\n" + "="*80)
    print("QUICK COMPARISON: BASELINE vs PREPROCESSED MODEL")
    print("="*80)
    
    print("\nBASELINE MODEL (No Preprocessing):")
    print(f"  Exact Accuracy: {baseline_metrics['accuracy_exact']:.2%}")
    print(f"  Accuracy within ±1: {baseline_metrics['accuracy_within_one']:.2%}")
    print(f"  RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"  R² Score: {baseline_metrics['r2']:.4f}")
    
    if preprocessed_metrics:
        print("\nPREPROCESSED MODEL:")
        print(f"  Exact Accuracy: {preprocessed_metrics['accuracy_exact']:.2%}")
        print(f"  Accuracy within ±1: {preprocessed_metrics['accuracy_within_one']:.2%}")
        print(f"  RMSE: {preprocessed_metrics['rmse']:.4f}")
        print(f"  R² Score: {preprocessed_metrics['r2']:.4f}")
        
        print("\n" + "-"*80)
        print("IMPROVEMENT FROM PREPROCESSING:")
        print("-"*80)
        
        # Calculate improvements
        acc_improvement = preprocessed_metrics['accuracy_exact'] - baseline_metrics['accuracy_exact']
        acc_within_one_improvement = preprocessed_metrics['accuracy_within_one'] - baseline_metrics['accuracy_within_one']
        rmse_improvement = baseline_metrics['rmse'] - preprocessed_metrics['rmse']  # Negative is improvement
        r2_improvement = preprocessed_metrics['r2'] - baseline_metrics['r2']
        
        print(f"  Exact Accuracy: {acc_improvement:+.2%} improvement")
        print(f"  Accuracy within ±1: {acc_within_one_improvement:+.2%} improvement")
        print(f"  RMSE: {rmse_improvement:+.4f} (negative means improvement)")
        print(f"  R² Score: {r2_improvement:+.4f} improvement")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        if acc_improvement > 0:
            print("✓ Preprocessing improved model accuracy")
        else:
            print("✗ Preprocessing did not improve model accuracy")
            
        if rmse_improvement > 0:
            print("✓ Preprocessing reduced prediction error (RMSE)")
        else:
            print("✗ Preprocessing did not reduce prediction error")
    else:
        print("\nPreprocessed model metrics not provided.")
        print("Run the preprocessed model training to get comparison data.")

# Execute baseline model training
if __name__ == "__main__":
    print("Training Baseline Model (No Preprocessing)...")
    baseline_results = main_baseline()
    
    if baseline_results:
        print("\n" + "="*80)
        print("BASELINE MODEL TRAINING COMPLETE")
        print("="*80)
        print("To compare with preprocessed model:")
        print("1. Run your preprocessed model training")
        print("2. Use the quick_comparison() function with both sets of metrics")
        
        # Example of how to use quick_comparison (uncomment and modify when you have preprocessed metrics)
        """
        # Assuming you have preprocessed_metrics from your preprocessed model
        preprocessed_metrics = {
            'accuracy_exact': 0.65,  # Replace with actual value
            'accuracy_within_one': 0.90,  # Replace with actual value
            'rmse': 0.55,  # Replace with actual value
            'r2': 0.45  # Replace with actual value
        }
        
        quick_comparison(baseline_results['test_metrics'], preprocessed_metrics)
        """