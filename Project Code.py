import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
file_path = '/kaggle/input/healthcare-dataset/healthcare_dataset.csv'
data = pd.read_csv(file_path)

# Rename Billing Amount column
data.rename(columns={"Billing Amount": "insurance_cost"}, inplace=True)

# Drop irrelevant columns
drop_columns = [
    "Name", "Date of Admission", "Discharge Date", "Doctor", "Hospital", 
    "Insurance Provider", "Room Number"
]
data.drop(columns=drop_columns, inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
data['Medical Condition'] = label_enc.fit_transform(data['Medical Condition'])
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Admission Type'] = label_enc.fit_transform(data['Admission Type'])

# Define severity grading function
def grade_severity(test_result):
    test_result = test_result.strip().lower()
    return {"normal": 0, "inconclusive": 1, "abnormal": 2}.get(test_result, -1)

# Apply severity grading
data['severity'] = data['Test Results'].apply(grade_severity)

# Drop invalid rows
data = data[data['severity'] != -1]

# Remove negative billing amounts
data = data[data['insurance_cost'] > 0]

# Convert "insurance_payment" into classification categories
def classify_payment(amount):
    if amount < 5000:
        return 0  # Low
    elif 5000 <= amount < 15000:
        return 1  # Medium
    else:
        return 2  # High

# Apply classification
data['insurance_category'] = data['insurance_cost'].apply(classify_payment)

# Prepare features and target
X = data[['Age', 'Medical Condition', 'Gender', 'Admission Type', 'severity']]
y = data['insurance_category']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define classification models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel='rbf', C=1.0, random_state=42)  # SVM with RBF kernel
}

# Store results
metrics = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store metrics
    metrics["Model"].append(model_name)
    metrics["Accuracy"].append(round(accuracy, 3))
    metrics["Precision"].append(round(precision, 3))
    metrics["Recall"].append(round(recall, 3))
    metrics["F1-Score"].append(round(f1, 3))

# Create DataFrame for results
metrics_df = pd.DataFrame(metrics)

# Display table
print("\n Group - 8 \nModel Performance Metrics:\n")
print(metrics_df.to_string(index=False))

# Identify the best model based on highest accuracy
best_model = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]
print(f"\n Best Model based on Accuracy: {best_model}")

# Confusion Matrix for best model
best_clf = models[best_model]
y_pred_best = best_clf.predict(X_test)

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, cmap="Blues", fmt="d",
            xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model}")
plt.show()