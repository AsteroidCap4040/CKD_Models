import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import joblib

# Define paths to your models and dataset
MODEL_PATHS = {
    'random_forest': 'models/random_forest_model.pkl',
    'naive_bayes': 'models/naive_bayes_model.pkl',
    'decision_tree': 'models/decision_tree_model.pkl',
    'xgboost': 'models/xgboost_model.pkl'
}

# Load the cleaned dataset
df = pd.read_excel('cleaned_ckd_data.xlsx')

# Drop rows where the target column 'classification' is missing
df = df.dropna(subset=['classification'])

# Separate features (X) and target (y)
X = df.drop('classification', axis=1)
y = df['classification']

# Apply imputation for any missing values in the features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)  # Impute missing values in X

# Check for remaining missing values after imputation
print("Number of missing values in X after imputation:", pd.DataFrame(X).isna().sum().sum())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the individual models
models = {}
for model_name, model_path in MODEL_PATHS.items():
    models[model_name] = joblib.load(model_path)

# Create the VotingClassifier using soft voting to enable probability predictions
voting_clf = VotingClassifier(estimators=[
    ('random_forest', models['random_forest']),
    ('naive_bayes', models['naive_bayes']),
    ('decision_tree', models['decision_tree']),
    ('xgboost', models['xgboost'])
], voting='soft')  # Soft voting for probability prediction

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Make predictions with the trained Voting Classifier
y_pred = voting_clf.predict(X_test)

# Evaluate the ensemble model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Voting Classifier Accuracy: {accuracy*100:.2f}%")

# More detailed evaluation (Confusion Matrix and Classification Report)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained Voting Classifier model
joblib.dump(voting_clf, 'voting_classifier_model.pkl')
print("✅ Voting Classifier model saved as 'voting_classifier_model.pkl'")

