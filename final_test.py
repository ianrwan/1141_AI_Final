# Wine Quality Prediction with XGBoost Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
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

# 2. Train XGBoost Model (version compatible)
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model"""
    print("\n=== Training XGBoost Model ===")
    
    # Initialize XGBoost regressor
    print("Initializing XGBoost regression model...")
    
    # Try different version approaches
    try:
        # Method 1: New version approach (with early_stopping_rounds)
        model = xgb.XGBRegressor(
            n_estimators=300,           # Number of trees
            learning_rate=0.05,         # Learning rate
            max_depth=5,                # Maximum tree depth
            min_child_weight=1,         # Minimum child weight
            subsample=0.8,              # Sample ratio per tree
            colsample_bytree=0.8,       # Feature ratio per tree
            random_state=42,
            n_jobs=-1,                  # Use all CPU cores
            eval_metric='rmse'          # Evaluation metric
        )
        
        print("Training with new XGBoost approach...")
        # Use eval_set to monitor training
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,  # Stop if no improvement for 20 rounds
            verbose=50                 # Output every 50 rounds
        )
        
    except TypeError as e:
        # Method 2: Old version approach (without early_stopping_rounds)
        print(f"New version approach failed: {e}")
        print("Trying old XGBoost approach...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,           # Reduce trees to avoid overfitting
            learning_rate=0.1,          # Increase learning rate
            max_depth=5,                # Maximum tree depth
            min_child_weight=1,         # Minimum child weight
            subsample=0.8,              # Sample ratio per tree
            colsample_bytree=0.8,       # Feature ratio per tree
            random_state=42,
            n_jobs=-1                   # Use all CPU cores
        )
        
        # Old version doesn't have early_stopping_rounds parameter
        model.fit(X_train, y_train)
    
    # Predict on training set
    y_train_pred = model.predict(X_train)
    
    # Predict on testing set
    y_test_pred = model.predict(X_test)
    
    # Round predictions to nearest integer (since quality scores are integers)
    y_test_pred_rounded = np.round(y_test_pred).astype(int)
    
    return model, y_train_pred, y_test_pred, y_test_pred_rounded

# 3. Evaluate Model Performance
def evaluate_model(y_true, y_pred, dataset_name="Dataset"):
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
    
    # For wine quality prediction, also calculate exact accuracy
    # Since quality is integer, we can round and compare
    y_pred_rounded = np.round(y_pred).astype(int)
    accuracy_exact = np.mean(y_true == y_pred_rounded)
    
    # Calculate prediction within ±1 range (close prediction)
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

# 4. Visualize Model Results (corrected version)
def visualize_model_results(y_test, y_test_pred, y_test_pred_rounded, model, X_train):
    """Visualize model prediction results and feature importance"""
    print("\n=== Visualizing Model Results ===")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Actual vs Predicted Scatter Plot
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction Line')
    axes[0, 0].set_xlabel('Actual Quality Score')
    axes[0, 0].set_ylabel('Predicted Quality Score')
    axes[0, 0].set_title('Actual vs Predicted (Test Set)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Prediction Error Distribution
    residuals = y_test - y_test_pred
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance Bar Chart
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
    axes[0, 2].set_title('XGBoost Feature Importance')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    # 4. Confusion Matrix Style Plot (showing rounded predictions vs actual)
    # Ensure all classes are present
    all_classes = np.arange(min(y_test.min(), y_test_pred_rounded.min()), 
                           max(y_test.max(), y_test_pred_rounded.max()) + 1)
    
    cm = confusion_matrix(y_test, y_test_pred_rounded, labels=all_classes)
    
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Prediction vs Actual (Rounded)')
    axes[1, 0].set_xlabel('Predicted Quality')
    axes[1, 0].set_ylabel('Actual Quality')
    
    # Display values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                          horizontalalignment="center",
                          color="white" if cm[i, j] > thresh else "black")
    
    # Set tick labels
    axes[1, 0].set_xticks(range(len(all_classes)))
    axes[1, 0].set_yticks(range(len(all_classes)))
    axes[1, 0].set_xticklabels(all_classes)
    axes[1, 0].set_yticklabels(all_classes)
    
    # 5. Prediction Accuracy by Quality Score
    quality_scores = np.unique(y_test)
    accuracy_by_quality = []
    
    for score in quality_scores:
        mask = y_test == score
        if mask.any():
            # Calculate prediction accuracy for this quality score (rounded)
            accuracy = np.mean(y_test_pred_rounded[mask] == score)
            accuracy_by_quality.append(accuracy)
    
    axes[1, 1].bar(quality_scores, accuracy_by_quality, alpha=0.7)
    axes[1, 1].set_xlabel('Wine Quality Score')
    axes[1, 1].set_ylabel('Prediction Accuracy')
    axes[1, 1].set_title('Prediction Accuracy by Quality Score')
    axes[1, 1].set_xticks(quality_scores)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. XGBoost Learning Curve (using eval_set results)
    try:
        # Try to get evaluation results
        results = model.evals_result()
        if results:
            # Get evaluation metrics
            train_rmse = results['validation_0']['rmse']
            test_rmse = results['validation_1']['rmse']
            
            epochs = len(train_rmse)
            axes[1, 2].plot(range(epochs), train_rmse, label='Training RMSE')
            axes[1, 2].plot(range(epochs), test_rmse, label='Testing RMSE')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('RMSE')
            axes[1, 2].set_title('XGBoost Learning Curve')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            # Mark best iteration
            best_iteration = model.best_iteration
            if best_iteration is not None:
                axes[1, 2].axvline(x=best_iteration, color='r', linestyle='--', alpha=0.5)
                axes[1, 2].text(best_iteration, test_rmse[best_iteration], 
                               f'  Best Iteration: {best_iteration}', 
                               verticalalignment='bottom')
        else:
            raise ValueError("No evaluation results")
    except Exception as e:
        # If no evaluation results, show feature importance table
        print(f"Cannot plot learning curve: {e}")
        axes[1, 2].axis('off')
        top_features = importance_df.head(8)
        feature_text = "Top 8 Feature Importance:\n\n"
        for _, row in top_features.iterrows():
            feature_text += f"{row.feature}: {row.importance:.4f}\n"
        
        axes[1, 2].text(0.5, 0.5, feature_text,
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=10,
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Feature Importance Ranking')
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

# 5. Save Model
def save_model(model, model_name='xgboost_wine_quality_model.json'):
    """Save trained XGBoost model"""
    print(f"\nSaving model to {model_name}...")
    model.save_model(model_name)
    print("Model saved successfully!")

# 6. Analyze Feature Importance in Detail
def analyze_feature_importance(importance_df, X_train, y_train):
    """Deep analysis of feature importance"""
    print("\n=== Deep Analysis of Feature Importance ===")
    
    # Show top 10 most important features
    print("\nFeature Importance Ranking (Top 10):")
    print(importance_df.head(10).to_string(index=False))
    
    # Calculate total feature importance
    total_importance = importance_df['importance'].sum()
    print(f"\nTotal Feature Importance: {total_importance:.4f}")
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    
    # Find number of features reaching 80% importance
    threshold = 0.8
    n_features_80 = (importance_df['cumulative_importance'] >= threshold).idxmax() + 1
    print(f"Number of features reaching {threshold:.0%} cumulative importance: {n_features_80}")
    
    # Plot cumulative importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance_df)), importance_df['cumulative_importance'])
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'{threshold:.0%} Threshold')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Feature Cumulative Importance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return importance_df

# 7. Main Program
def main():
    """Main program: Execute complete modeling pipeline"""
    print("=" * 60)
    print("Wine Quality Prediction - XGBoost Model Training")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        return
    
    # Train model
    model, y_train_pred, y_test_pred, y_test_pred_rounded = train_xgboost_model(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Model Performance Evaluation")
    print("=" * 60)
    
    print("Training Set Evaluation:")
    train_metrics = evaluate_model(y_train, y_train_pred, "Training Set")
    
    print("\nTesting Set Evaluation:")
    test_metrics = evaluate_model(y_test, y_test_pred, "Testing Set")
    
    # Show sample prediction results
    print("\nTesting Set - First 10 Sample Predictions:")
    results_df = pd.DataFrame({
        'Actual Quality': y_test[:10],
        'Predicted Quality(Raw)': np.round(y_test_pred[:10], 2),
        'Predicted Quality(Rounded)': y_test_pred_rounded[:10]
    })
    print(results_df)
    
    # Visualize results
    importance_df = visualize_model_results(y_test, y_test_pred, y_test_pred_rounded, model, X_train)
    
    # Deep analysis of feature importance
    importance_df = analyze_feature_importance(importance_df, X_train, y_train)
    
    # Save model
    save_model(model)
    
    # Summary report
    print("\n" + "=" * 60)
    print("Modeling Completion Summary")
    print("=" * 60)
    
    # Try to get best iteration
    try:
        best_iteration = model.best_iteration
        print(f"1. Best Iteration: {best_iteration}")
    except:
        print(f"1. Best Iteration: Not available")
    
    print(f"2. Training Set Exact Accuracy: {train_metrics['accuracy_exact']:.2%}")
    print(f"3. Testing Set Exact Accuracy: {test_metrics['accuracy_exact']:.2%}")
    print(f"4. Testing Set Accuracy within ±1: {test_metrics['accuracy_within_one']:.2%}")
    print(f"5. Most Important Feature: {importance_df.iloc[0]['feature']} "
          f"(Importance: {importance_df.iloc[0]['importance']:.4f})")
    print(f"6. Model saved as 'xgboost_wine_quality_model.json'")
    
    # Provide improvement suggestions
    print("\n=== Improvement Suggestions ===")
    if test_metrics['accuracy_exact'] < 0.5:
        print("Model accuracy is low, suggestions:")
        print("  - Try adjusting XGBoost hyperparameters (increase n_estimators, adjust learning_rate)")
        print("  - Use grid search or random search for hyperparameter optimization")
        print("  - Try other models like Random Forest, LightGBM")
        print("  - Review feature engineering, may need more wine chemistry domain knowledge")
    elif test_metrics['accuracy_exact'] < 0.7:
        print("Model accuracy is moderate, suggestions:")
        print("  - Perform hyperparameter tuning (use GridSearchCV or RandomizedSearchCV)")
        print("  - Try ensemble multiple models (e.g., XGBoost + Random Forest)")
        print("  - Add more wine chemistry related features")
        print("  - Try treating problem as classification rather than regression")
    else:
        print("Model accuracy is good! Suggestions:")
        print("  - Consider deploying model for practical applications")
        print("  - Further optimize for higher accuracy")
        print("  - Consider using more complex ensemble methods")
    
    return model, test_metrics, importance_df

# Execute main program
if __name__ == "__main__":
    model, metrics, importance_df = main()