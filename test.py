import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv('data.csv')

# Check for missing values and display the count of null values in each column
print("Missing values in each column:")
print(df.isnull().sum())

# Prepare the features and target variable
X = df.drop('Glucose', axis=1)  # Drop 'Glucose' from features
y = df['Glucose']  # Target variable is 'Glucose'

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Initialize and train the XGBoost Regressor model
model = XGBRegressor(n_estimators=100, random_state=32)
model.fit(X_train, y_train)

# Predict the blood glucose levels on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Absolute Error and Root Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Plot the actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='red', label='Predicted vs Actual')
plt.xlabel('Actual Blood Glucose Level')
plt.ylabel('Predicted Blood Glucose Level')
plt.title('Actual vs Predicted Blood Glucose Levels')
plt.grid(True)
plt.show()

# Save the trained model to a file using joblib
#joblib.dump(model, 'xgboost_model.pkl')  # Save the model as 'xgboost_model.pkl'
#print("Model saved to 'xgboost_model.pkl'")

# Load the saved model from the file
loaded_model = joblib.load('xgboost_model.pkl')
print("Model loaded from 'xgboost_model.pkl'")

# Example of making predictions with the loaded model (can be used for Flask API or other applications)
example_input = X_test.iloc[0:1]  # Taking a single sample from the test set as an example
example_prediction = loaded_model.predict(example_input)
print(f'Example prediction for the input: {example_prediction}')
