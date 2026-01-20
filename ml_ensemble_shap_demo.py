# 필요 라이브러리 설치 (코랩 사용 시)
# !pip install xgboost shap

import pandas as pd
import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 데이터 준비 (Scikit-learn & Useful Things to Know)
# 데이터를 훈련용과 테스트용으로 나누는 것이 '일반화'의 시작입니다.
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 60)
print("[DATA] Iris Dataset Info")
print("=" * 60)
print(f"전체 샘플 수: {len(X)}")
print(f"훈련 데이터: {len(X_train)} 샘플")
print(f"테스트 데이터: {len(X_test)} 샘플")
print(f"특징 변수: {list(X.columns)}")
print(f"클래스: {iris.target_names}")
print()

# 2. Random Forest 실행 (Ensemble 기법)
print("=" * 60)
print("[RF] Random Forest Training...")
print("=" * 60)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Random Forest Feature Importance
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Random Forest Feature Importance:")
for idx, row in rf_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
print()

# 3. XGBoost 실행 (Scalable Boosting)
print("=" * 60)
print("[XGB] XGBoost Training...")
print("=" * 60)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print()

# 4. 결과 출력 (Preview)
print("=" * 60)
print("[RESULT] Model Performance Report")
print("=" * 60)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"Random Forest 정확도: {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print(f"XGBoost 정확도:       {xgb_acc:.4f} ({xgb_acc*100:.1f}%)")
print()

# 5. SHAP를 이용한 모델 해석 (Interpretability)
# 모델이 어떤 특징(꽃잎 길이 등)을 보고 판단했는지 시각화합니다.
print("=" * 60)
print("[SHAP] Analyzing with SHAP...")
print("=" * 60)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# SHAP 값 요약 출력
print("\n[SHAP Feature Importance - 각 특징이 예측에 미친 평균 영향력]")
if isinstance(shap_values, list):
    # Multi-class의 경우 평균
    mean_shap = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': [abs(shap_values[0][:, i]).mean() + abs(shap_values[1][:, i]).mean() + abs(shap_values[2][:, i]).mean() for i in range(len(X.columns))]
    }).sort_values('mean_abs_shap', ascending=False)
else:
    mean_shap = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': [abs(shap_values[:, i]).mean() for i in range(len(X.columns))]
    }).sort_values('mean_abs_shap', ascending=False)

for idx, row in mean_shap.iterrows():
    bar = "#" * int(row['mean_abs_shap'] * 20)
    print(f"  {row['feature']}: {row['mean_abs_shap']:.4f} {bar}")

print("\n" + "=" * 60)
print("[PLOT] Generating SHAP Summary Plot...")
print("=" * 60)

# SHAP Summary Plot 저장
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
print("[OK] SHAP Summary Plot saved as 'shap_summary_plot.png'")

print("\n" + "=" * 60)
print("[DONE] Analysis Complete!")
print("=" * 60)
