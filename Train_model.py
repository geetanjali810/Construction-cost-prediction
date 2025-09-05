import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Read CSV
df = pd.read_csv("construction_costs.csv")

# If total_cost is placeholder 0, auto-generate realistic values
if df['total_cost'].sum() == 0:
    np.random.seed(42)
    base = (df['area_m2'] * (df['material_cost_per_m2'] + df['labor_cost_per_m2']) * df['num_floors'] * 0.6)
    df['total_cost'] = base + df['equipment_cost'] + (df['location_factor'] * 10000) + np.random.normal(0, base*0.05, size=len(df))

X = df[['area_m2','num_floors','material_cost_per_m2','labor_cost_per_m2','equipment_cost','location_factor']]
X = pd.concat([X, pd.get_dummies(df['project_type'], prefix='type', drop_first=True)], axis=1)
y = df['total_cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2:", r2_score(y_test, y_pred))

fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nFeature importances:\n", fi)

# Save trained model
joblib.dump({"model": model, "columns": X_train.columns.tolist()}, "construction_cost_model.joblib")
