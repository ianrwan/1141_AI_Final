# Wine Quality Prediction with XGBoost - Simplified Version
# This file is: final_test.py (Simplified for clear metrics)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1. Load Preprocessed Data
def load_preprocessed_data():
    """Load preprocessed training and testing data"""
    print("Loading preprocessed data...")
    
    try:
        X_train = pd.read_csv('X_train_preprocessed.csv')
        X_test = pd.read_csv('X_test_preprocessed.csv')
        y_train = pd.read_csv('y_train_preprocessed.csv')
        y_test = pd.read_csv('y_test_preprocessed.csv')
        
        # Check if DataFrame and extract arrays
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values.ravel()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.ravel()
        
        print(f"Training data shape: X_train={X_train.shape}, y_train={len(y_train)}")
        print(f"Testing data shape: X_test={X_test.shape}, y_test={len(y_test)}")
        
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please run the preprocessing pipeline first to generate data files")
        return None, None, None, None

# 2. Train XGBoost Model
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model"""
    print("\n=== Training XGBoost Model ===")
    
    # Initialize XGBoost regressor
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
    
    return model, y_train_pred, y_test_pred

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

# 4. Simple Visualization
def simple_visualization(y_test, y_test_pred, model, X_train):
    """Simple visualization of key results"""
    print("\n=== Simple Visualization ===")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Quality')
    axes[0].set_ylabel('Predicted Quality')
    axes[0].set_title('Actual vs Predicted (Test Set)')
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
    axes[1].set_title('Top 10 Feature Importance')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

# 5. Main Program - SIMPLIFIED
def main():
    """Main program: Simplified version"""
    print("=" * 60)
    print("Wine Quality Prediction - XGBoost Model")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        return
    
    # Train model
    model, y_train_pred, y_test_pred = train_xgboost_model(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE - TESTING SET")
    print("=" * 60)
    
    test_metrics = evaluate_model(y_test, y_test_pred, "Testing Set")
    
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
    importance_df = simple_visualization(y_test, y_test_pred, model, X_train)
    
    # Save model
    print("\nSaving model...")
    model.save_model('xgboost_wine_quality_model.json')
    print("Model saved as 'xgboost_wine_quality_model.json'")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TESTING SET RESULTS SUMMARY")
    print("=" * 60)
    print(f"Exact match accuracy: {test_metrics['accuracy_exact']:.2%}")
    print(f"Accuracy within ±1 score: {test_metrics['accuracy_within_one']:.2%}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    
    # Performance assessment
    print("\n" + "-" * 60)
    print("PERFORMANCE ASSESSMENT:")
    
    if test_metrics['accuracy_within_one'] >= 0.85:
        print("✅ EXCELLENT - Meets industry standard (≥85% within ±1)")
    elif test_metrics['accuracy_within_one'] >= 0.75:
        print("⚠️ ACCEPTABLE - Good but could improve (75-85% within ±1)")
    else:
        print("❌ NEEDS IMPROVEMENT - Below expectations (<75% within ±1)")
    
    return model, test_metrics, importance_df

# Execute main program
if __name__ == "__main__":
    model, metrics, importance_df = main()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)