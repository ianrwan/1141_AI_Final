# Wine Quality Prediction with XGBoost - Raw Data Version
# This file is: final_test_raw.py (Using raw data without preprocessing)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1. Load Raw Data
def load_raw_data():
    """Load raw wine quality data"""
    print("Loading raw wine quality data...")
    
    try:
        # Note: Data uses semicolon delimiter
        df = pd.read_csv('dataset/winequality-red.csv', sep=';')
        print(f"Raw data loaded successfully! Total {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Prepare data - separate features and target
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Features used: {list(X.columns)}")
        
        return X, y
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None

# 2. Train XGBoost Model on Raw Data
def train_xgboost_model_raw(X, y):
    """Train and evaluate XGBoost model on raw data"""
    print("\n=== Training XGBoost Model on RAW DATA ===")
    
    # Split data (same parameters as preprocessed version)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Initialize XGBoost regressor (same parameters as preprocessed version)
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

# 3. Evaluate Model Performance - SIMPLIFIED VERSION
def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
    """Evaluate model performance - focusing on key metrics"""
    print(f"\n=== {dataset_name} Evaluation ===")
    
    # Calculate standard regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Round predictions for accuracy calculation
    y_pred_rounded = np.round(y_pred).astype(int)
    
    # Calculate exact match accuracy
    accuracy_exact = np.mean(y_true == y_pred_rounded)
    
    # Calculate accuracy within ±1 score
    accuracy_within_one = np.mean(np.abs(y_true - y_pred_rounded) <= 1)
    
    print(f"Exact match accuracy: {accuracy_exact:.2%}")
    print(f"Accuracy within ±1 score: {accuracy_within_one:.2%}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'accuracy_exact': accuracy_exact,
        'accuracy_within_one': accuracy_within_one,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

# 4. Simple Visualization for Raw Data
def simple_visualization_raw(y_test, y_test_pred, model, X_train):
    """Simple visualization of key results for raw data"""
    print("\n=== Simple Visualization for RAW DATA ===")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Quality')
    axes[0].set_ylabel('Predicted Quality')
    axes[0].set_title('Actual vs Predicted (Raw Data)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Feature Importance
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1].barh(range(len(importance_df)), importance_df['importance'])
    axes[1].set_yticks(range(len(importance_df)))
    axes[1].set_yticklabels(importance_df['feature'])
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Top 10 Feature Importance (Raw Data)')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

# 5. Compare Raw vs Preprocessed Results
def compare_results(raw_metrics, preprocessed_metrics=None):
    """Compare results between raw and preprocessed data"""
    print("\n" + "=" * 60)
    print("COMPARISON: RAW DATA vs PREPROCESSED DATA")
    print("=" * 60)
    
    print(f"\nRAW DATA Results:")
    print(f"  Exact match accuracy: {raw_metrics['accuracy_exact']:.2%}")
    print(f"  Accuracy within ±1 score: {raw_metrics['accuracy_within_one']:.2%}")
    print(f"  MAE: {raw_metrics['mae']:.4f}")
    print(f"  RMSE: {raw_metrics['rmse']:.4f}")
    print(f"  R² Score: {raw_metrics['r2']:.4f}")
    
    if preprocessed_metrics:
        print(f"\nPREPROCESSED DATA Results:")
        print(f"  Exact match accuracy: {preprocessed_metrics['accuracy_exact']:.2%}")
        print(f"  Accuracy within ±1 score: {preprocessed_metrics['accuracy_within_one']:.2%}")
        print(f"  MAE: {preprocessed_metrics['mae']:.4f}")
        print(f"  RMSE: {preprocessed_metrics['rmse']:.4f}")
        print(f"  R² Score: {preprocessed_metrics['r2']:.4f}")
        
        print(f"\n" + "-" * 60)
        print(f"IMPROVEMENT FROM PREPROCESSING:")
        print(f"-" * 60)
        
        # Calculate improvements
        acc_exact_improvement = preprocessed_metrics['accuracy_exact'] - raw_metrics['accuracy_exact']
        acc_within_one_improvement = preprocessed_metrics['accuracy_within_one'] - raw_metrics['accuracy_within_one']
        mae_improvement = raw_metrics['mae'] - preprocessed_metrics['mae']  # Negative is better
        rmse_improvement = raw_metrics['rmse'] - preprocessed_metrics['rmse']  # Negative is better
        r2_improvement = preprocessed_metrics['r2'] - raw_metrics['r2']
        
        print(f"  Exact match accuracy: {acc_exact_improvement:+.2%}")
        print(f"  Accuracy within ±1 score: {acc_within_one_improvement:+.2%}")
        print(f"  MAE improvement: {mae_improvement:+.4f} (negative means better)")
        print(f"  RMSE improvement: {rmse_improvement:+.4f} (negative means better)")
        print(f"  R² Score improvement: {r2_improvement:+.4f}")
        
        # Summary
        print(f"\nSUMMARY:")
        if acc_within_one_improvement > 0:
            print(f"✅ Preprocessing improved accuracy within ±1 score")
        else:
            print(f"⚠️ Preprocessing did not improve accuracy within ±1 score")
            
        if mae_improvement > 0:
            print(f"✅ Preprocessing reduced MAE (improved predictions)")
        else:
            print(f"⚠️ Preprocessing did not reduce MAE")

# 6. Main Program for Raw Data
def main_raw():
    """Main program for raw data analysis"""
    print("=" * 60)
    print("Wine Quality Prediction - RAW DATA (No Preprocessing)")
    print("=" * 60)
    
    # Load raw data
    X, y = load_raw_data()
    
    if X is None:
        return
    
    # Train model on raw data
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_xgboost_model_raw(X, y)
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE - RAW DATA (Testing Set)")
    print("=" * 60)
    
    test_metrics_raw = evaluate_model(y_test, y_test_pred, "Testing Set (Raw Data)")
    
    # Show sample predictions
    print("\n" + "-" * 60)
    print("Sample Predictions (First 10)")
    print("-" * 60)
    
    results_df = pd.DataFrame({
        'Actual': y_test[:10],
        'Predicted': np.round(y_test_pred[:10], 2),
        'Rounded': np.round(y_test_pred[:10]).astype(int),
        'Error': y_test[:10] - np.round(y_test_pred[:10]).astype(int)
    })
    print(results_df)
    
    # Simple visualization
    importance_df = simple_visualization_raw(y_test, y_test_pred, model, X_train)
    
    # Save model
    print("\nSaving raw data model...")
    model.save_model('xgboost_wine_quality_model_raw.json')
    print("Model saved as 'xgboost_wine_quality_model_raw.json'")
    
    # Final summary
    print("\n" + "=" * 60)
    print("RAW DATA TESTING SET RESULTS SUMMARY")
    print("=" * 60)
    print(f"Exact match accuracy: {test_metrics_raw['accuracy_exact']:.2%}")
    print(f"Accuracy within ±1 score: {test_metrics_raw['accuracy_within_one']:.2%}")
    print(f"MAE: {test_metrics_raw['mae']:.4f}")
    print(f"RMSE: {test_metrics_raw['rmse']:.4f}")
    
    # Performance assessment
    print("\n" + "-" * 60)
    print("PERFORMANCE ASSESSMENT (Raw Data):")
    
    if test_metrics_raw['accuracy_within_one'] >= 0.85:
        print("✅ EXCELLENT - Meets industry standard (≥85% within ±1)")
    elif test_metrics_raw['accuracy_within_one'] >= 0.75:
        print("⚠️ ACCEPTABLE - Good but could improve (75-85% within ±1)")
    else:
        print("❌ NEEDS IMPROVEMENT - Below expectations (<75% within ±1)")
    
    return model, test_metrics_raw, importance_df

# 7. Execute with option to compare with preprocessed results
def main():
    """Main execution with optional comparison"""
    print("Starting Raw Data Analysis...")
    print("This will train XGBoost on raw data without any preprocessing.")
    print("\nNote: Run the preprocessed version first to get comparison data.")
    
    # Run raw data analysis
    model_raw, metrics_raw, importance_df_raw = main_raw()
    
    # Ask if user wants to compare with preprocessed results
    print("\n" + "=" * 60)
    response = input("Do you have preprocessed results to compare? (y/n): ")
    
    if response.lower() == 'y':
        # These would be the metrics from your preprocessed model
        # You'll need to run the preprocessed version first and note down the metrics
        print("\nPlease enter your preprocessed model metrics:")
        
        try:
            preprocessed_accuracy_exact = float(input("Exact match accuracy (e.g., 0.625 for 62.5%): "))
            preprocessed_accuracy_within_one = float(input("Accuracy within ±1 score (e.g., 0.9125 for 91.25%): "))
            preprocessed_mae = float(input("MAE (e.g., 0.4562): "))
            preprocessed_rmse = float(input("RMSE (e.g., 0.5893): "))
            preprocessed_r2 = float(input("R² Score (e.g., 0.3245): "))
            
            preprocessed_metrics = {
                'accuracy_exact': preprocessed_accuracy_exact,
                'accuracy_within_one': preprocessed_accuracy_within_one,
                'mae': preprocessed_mae,
                'rmse': preprocessed_rmse,
                'r2': preprocessed_r2
            }
            
            # Compare results
            compare_results(metrics_raw, preprocessed_metrics)
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    else:
        print("\nTo compare results:")
        print("1. Run the preprocessed model (final_test.py)")
        print("2. Note down the testing set metrics")
        print("3. Run this script again and enter those metrics when prompted")
    
    print("\n" + "=" * 60)
    print("RAW DATA ANALYSIS COMPLETE")
    print("=" * 60)
    
    return model_raw, metrics_raw, importance_df_raw

# Execute main program
if __name__ == "__main__":
    model_raw, metrics_raw, importance_df_raw = main()