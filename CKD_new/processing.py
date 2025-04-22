import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('kidney_disease(1).csv')

# Drop 'id' column if it exists
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Convert numeric columns properly
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
# Fill numeric columns with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical text into binary
binary_mapping = {
    'normal': 0, 'abnormal': 1,
    'present': 1, 'notpresent': 0,
    'yes': 1, 'no': 0,
    'good': 0, 'poor': 1,
    'ckd': 1, 'notckd': 0
}

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_mapping)

# Now scale only specific columns (not age, bp, sg, al, su)
scale_cols = ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Save the cleaned dataset to a new Excel file
df.to_excel('cleaned_ckd_data.xlsx', index=False)

print("✅ Data cleaned properly and saved to 'cleaned_ckd_data.xlsx'")
