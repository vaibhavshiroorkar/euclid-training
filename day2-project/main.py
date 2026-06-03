import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset

df = pd.read_csv('gym_members_exercise_tracking.csv')
print("Dataset loaded successfully")

print("Shape", df.shape)
df.head()

# Step 2: Explore the Data

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df[['Session_Duration (hours)', 'Calories_Burned']].describe())

correlation = df['Session_Duration (hours)'].corr(df['Calories_Burned'])
print(f"\nCorrelation between Duration and Calories: {correlation:.4f}")

# Graph for Plotting

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Session_Duration (hours)', y='Calories_Burned', alpha=0.5, color='orange')
plt.title('Workout Duration vs Calories Burned')
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.tight_layout()
plt.show()

# Step 3: Select the X and y

X = df[['Session_Duration (hours)']]
y = df['Calories_Burned']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Step 4: Split into 80% Training and 20% Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Step 5: Scale the features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")

# Step 6: Train the models

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
}

for name, model in models.items():
  model.fit(X_train_scaled, y_train)
  print(f"{name} --> Trained")

# Step 7: Evaluate the models

results = {}

print(f"{'Model':<28} {'MAE':>8} {'RMSE':>10} {'R2 Score':>10}")
print("-" * 60)

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"{name:<28} {mae:>8.2f} {rmse:>10.2f} {r2:>10.4f}")

# Step 8: Compare results and choose the best one

results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)

best_model_name = results_df.index[0]
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"  R2 Score : {results_df.loc[best_model_name, 'R2']:.4f}")
print(f"  MAE      : {results_df.loc[best_model_name, 'MAE']:.2f}")
print(f"  RMSE     : {results_df.loc[best_model_name, 'RMSE']:.2f}")

# Bar Chart for Comparison

results_df['R2'].plot(kind='barh', color='steelblue', figsize=(9, 5))
plt.title('Model Comparison - R2 Score')
plt.xlabel('R2 Score')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# Step 9: Predict on new data

new_duration = pd.DataFrame([[1.5]], columns=['Session_Duration (hours)'])
new_duration_scaled = scaler.transform(new_duration)

predicted_calories = best_model.predict(new_duration_scaled)

print(f"For a workout duration of {new_duration.iloc[0, 0]} hours,")
print(f"the {best_model_name} predicts approximately {int(predicted_calories[0])} calories burned.")