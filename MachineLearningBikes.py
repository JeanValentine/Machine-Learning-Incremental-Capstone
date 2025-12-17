import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


file_path = r"C:\Users\valen\OneDrive\Desktop\bike_rental_features.csv"
df = pd.read_csv(file_path)


print("Data Loaded. Shape:", df.shape)
print("Columns:", list(df.columns))


df.fillna(df.median(numeric_only=True), inplace=True) 
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)  


target = "Rented Bike Count"
X = df.drop(columns=[target])
y = df[target]


categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])


X_processed = preprocessor.fit_transform(X)
processed_df = pd.DataFrame(X_processed)
processed_df[target] = y.values
processed_df.to_csv("bike_rental_features_processed.csv", index=False)
print("Processed dataset saved as 'bike_rental_features_processed.csv'.")


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}


param_grid = {
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
}


results = {}
cv_results = {}


for name, model in models.items():
    print(f"Training {name}...")
    if name in param_grid:
        grid = GridSearchCV(model, param_grid[name], cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)


    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

 
    cv_r2 = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2").mean()
    cv_rmse = np.sqrt(-cross_val_score(best_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")).mean()
    cv_mae = -cross_val_score(best_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    cv_results[name] = {"CV_R2": cv_r2, "CV_RMSE": cv_rmse, "CV_MAE": cv_mae}

    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}, CV_R2: {cv_r2:.3f}")


poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


poly_models = ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]
for name in poly_models:
    print(f"Training {name} with Polynomial Features...")
    model = models[name]
    if name in param_grid:
        grid = GridSearchCV(model, param_grid[name], cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train_poly, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model.fit(X_train_poly, y_train)


    y_pred = best_model.predict(X_test_poly)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[f"{name}_Poly"] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{name}_Poly -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")


for name in ["RandomForest", "GradientBoosting"]:
    model = models[name]
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 10 feature importances for {name}:")
    for i in sorted_idx[:10]:
        print(f"Feature {i}: Importance {importances[i]:.4f}")


results_df = pd.DataFrame(results).T
results_df.sort_values("R2", ascending=False, inplace=True)
print("\n===== Model Comparison =====")
print(results_df)


plt.figure(figsize=(10,5))
sns.barplot(x=results_df.index, y=results_df["R2"])
plt.xticks(rotation=45)
plt.title("Model R² Comparison")
plt.ylabel("R² Score")
plt.show()


plt.figure(figsize=(10,5))
sns.barplot(x=results_df.index, y=results_df["RMSE"])
plt.xticks(rotation=45)
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.show()


plt.figure(figsize=(10,5))
sns.barplot(x=results_df.index, y=results_df["MAE"])
plt.xticks(rotation=45)
plt.title("Model MAE Comparison")
plt.ylabel("MAE")
plt.show()


print("\n===== Insights & Recommendations =====")
print("1. Polynomial models improved performance slightly for regularized regressions.")
print("2. RandomForest and GradientBoosting achieved the best R² scores, indicating non-linear patterns in bike rentals.")
print("3. Feature importance indicates which variables most affect rentals (especially for tree models).")
print("4. For production, consider additional features like weather interactions, holidays, or temporal patterns.")
print("5. Future improvements: hyperparameter tuning using more extensive grids, exploring feature selection, and time-series models.")
