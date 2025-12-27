import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. 載入預處理後的數據
print("="*50)
print("載入預處理後的數據")
print("="*50)

X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv')
y_test = pd.read_csv('y_test_preprocessed.csv')

# 確保 y 是 1D 數組 (LogisticRegression 需要)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"訓練集形狀: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"測試集形狀: X_test: {X_test.shape}, y_test: {y_test.shape}")

# 檢查類別分佈
print("\n訓練集目標變量分佈:")
print(pd.Series(y_train).value_counts().sort_index())
print("\n測試集目標變量分佈:")
print(pd.Series(y_test).value_counts().sort_index())

# 2. 邏輯回歸模型 - 基礎版本
print("\n" + "="*50)
print("邏輯回歸模型 - 基礎版本")
print("="*50)

# 初始化邏輯回歸模型
# 由於品質評分有3-8分，共6個類別，所以使用 multinomial
# 因為特徵已標準化，所以 penalty='l2' (正則化) 是合適的
base_lr = LogisticRegression(
    multi_class='multinomial',  # 多類別分類
    solver='lbfgs',            # 適合中小型數據集
    max_iter=1000,             # 增加迭代次數確保收斂
    random_state=42
)

# 訓練模型
print("訓練邏輯回歸模型...")
base_lr.fit(X_train, y_train)

# 預測
y_train_pred = base_lr.predict(X_train)
y_test_pred = base_lr.predict(X_test)

# 計算準確率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"訓練集準確率: {train_accuracy:.4f}")
print(f"測試集準確率: {test_accuracy:.4f}")

# 3. 邏輯回歸模型 - 帶正則化的調優版本
print("\n" + "="*50)
print("邏輯回歸模型 - 調優版本 (網格搜索)")
print("="*50)

# 定義參數網格
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 正則化強度的倒數，越小正則化越強
    'penalty': ['l2'],  # L2正則化
    'solver': ['lbfgs', 'newton-cg'],
    'max_iter': [1000, 2000]
}

# 初始化模型
lr_tuned = LogisticRegression(
    multi_class='multinomial',
    random_state=42
)

# 使用網格搜索進行調優
print("進行網格搜索調優...")
grid_search = GridSearchCV(
    estimator=lr_tuned,
    param_grid=param_grid,
    cv=5,  # 5折交叉驗證
    scoring='accuracy',
    n_jobs=-1,  # 使用所有可用的CPU核心
    verbose=1
)

grid_search.fit(X_train, y_train)

# 最佳參數和模型
print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳交叉驗證準確率: {grid_search.best_score_:.4f}")

# 使用最佳模型進行預測
best_lr = grid_search.best_estimator_
y_train_pred_best = best_lr.predict(X_train)
y_test_pred_best = best_lr.predict(X_test)

# 計算準確率
train_accuracy_best = accuracy_score(y_train, y_train_pred_best)
test_accuracy_best = accuracy_score(y_test, y_test_pred_best)

print(f"調優後訓練集準確率: {train_accuracy_best:.4f}")
print(f"調優後測試集準確率: {test_accuracy_best:.4f}")

# 4. 詳細評估
print("\n" + "="*50)
print("詳細模型評估")
print("="*50)

print("\n1. 最佳模型測試集分類報告:")
print(classification_report(y_test, y_test_pred_best))

print("\n2. 混淆矩陣 (測試集):")
conf_matrix = confusion_matrix(y_test, y_test_pred_best)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=[f'實際 {i}' for i in sorted(np.unique(y_test))],
    columns=[f'預測 {i}' for i in sorted(np.unique(y_test))]
)
print(conf_matrix_df)

# 計算每個類別的準確率
print("\n3. 各類別準確率:")
for class_label in sorted(np.unique(y_test)):
    # 找出該類別的樣本
    class_indices = y_test == class_label
    if sum(class_indices) > 0:  # 確保有樣本
        class_accuracy = accuracy_score(
            y_test[class_indices], 
            y_test_pred_best[class_indices]
        )
        print(f"  品質 {class_label}: {class_accuracy:.4f} (樣本數: {sum(class_indices)})")

# 5. 特徵重要性分析
print("\n" + "="*50)
print("特徵重要性分析 (基於係數)")
print("="*50)

# 邏輯回歸的係數可以解釋特徵重要性
# 注意：對於多類別分類，每個類別都有一組係數
coefficients = best_lr.coef_

# 獲取特徵名稱
feature_names = X_train.columns.tolist()

print("\n特徵對各類別的重要性 (絕對值前5名):")
for i, class_label in enumerate(sorted(np.unique(y_train))):
    # 獲取該類別的係數
    class_coef = coefficients[i]
    
    # 創建特徵-係數對的DataFrame
    coef_df = pd.DataFrame({
        '特徵': feature_names,
        '係數': class_coef,
        '絕對值': np.abs(class_coef)
    })
    
    # 按絕對值排序
    coef_df = coef_df.sort_values('絕對值', ascending=False)
    
    print(f"\n品質 {class_label} 最重要的特徵:")
    print(coef_df.head(5).to_string(index=False))

# 計算整體特徵重要性 (所有類別係數絕對值的平均)
print("\n整體特徵重要性 (所有類別平均):")
overall_importance = pd.DataFrame({
    '特徵': feature_names,
    '平均絕對係數': np.mean(np.abs(coefficients), axis=0)
})
overall_importance = overall_importance.sort_values('平均絕對係數', ascending=False)
print(overall_importance.to_string(index=False))

# 6. 模型比較總結
print("\n" + "="*50)
print("模型效能總結")
print("="*50)

results_summary = pd.DataFrame({
    '模型': ['基礎邏輯回歸', '調優邏輯回歸'],
    '訓練集準確率': [train_accuracy, train_accuracy_best],
    '測試集準確率': [test_accuracy, test_accuracy_best],
    '泛化差距': [train_accuracy - test_accuracy, train_accuracy_best - test_accuracy_best]
})

print(results_summary.to_string(index=False))

# 7. 保存最佳模型和預測結果
print("\n" + "="*50)
print("保存結果")
print("="*50)

# 保存預測結果
predictions_df = pd.DataFrame({
    '實際品質': y_test,
    '預測品質': y_test_pred_best,
    '預測正確': y_test == y_test_pred_best
})
predictions_df.to_csv('logistic_regression_predictions.csv', index=False)
print("預測結果已保存到 'logistic_regression_predictions.csv'")

# 保存最佳模型參數
import json
model_info = {
    'best_params': str(grid_search.best_params_),
    'best_score': float(grid_search.best_score_),
    'test_accuracy': float(test_accuracy_best),
    'feature_names': feature_names,
    'coefficients': best_lr.coef_.tolist(),
    'intercepts': best_lr.intercept_.tolist()
}

with open('logistic_regression_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("模型信息已保存到 'logistic_regression_model_info.json'")

print("\n" + "="*50)
print("邏輯回歸建模完成!")
print("="*50)