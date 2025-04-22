import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving the model

# Load the cleaned dataset
df = pd.read_excel('cleaned_ckd_data.xlsx')

# 🛠 Drop rows where 'classification' is missing
df = df.dropna(subset=['classification'])

# Separate features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy*100:.2f}%")

# More detailed evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/xgboost_model.pkl')
print("Model saved as 'xgboost_model.pkl'")



