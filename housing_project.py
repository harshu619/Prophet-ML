import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# Step 1: Load and Prepare the Dataset
# ---------------------------------------------------------
print("Loading dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# We select 3 key features: Median Income, House Age, Average Rooms
features = ['MedInc', 'HouseAge', 'AveRooms']
X = df[features]
y = df['MedHouseVal']  # Fixed column name for newer scikit-learn versions

print("\n--- Dataset Sample ---")
print(X.head())

# ---------------------------------------------------------
# Step 2: Split Data into Training and Testing Sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# Step 3: Initialize and Train the Model
# ---------------------------------------------------------
print("\nTraining the Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# ---------------------------------------------------------
# Step 4: Make Predictions and Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (Accuracy Metric): {r2:.4f}")

# ---------------------------------------------------------
# Step 5: Predict Price for a Custom House
# ---------------------------------------------------------
# Custom house: Median Income = 5.0 ($50k), House Age = 25 years, Ave Rooms = 6
custom_house = pd.DataFrame([[5.0, 25, 6]], columns=features)
predicted_value = model.predict(custom_house)
print("\n--- Custom Prediction ---")
print(f"Predicted Custom House Value: ${predicted_value[0] * 100000:,.2f}")

# ---------------------------------------------------------
# Step 6: Visualize the Results
# ---------------------------------------------------------
print("\nGenerating graph... (Close the graph window to finish the script)")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Prices ($100k)')
plt.ylabel('Predicted Prices ($100k)')
plt.title('Actual vs Predicted House Prices')
plt.show()
