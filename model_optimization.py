import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
import joblib


# Load dataset
df = pd.read_csv("C:\\Users\\karsh\\OneDrive\\Desktop\\intern-performance-prediction\\intern_dataset (1).csv")

def assign_label(row):
    completion_rate = row['completion_rate']
    engagement = row['engagement_score']
    evaluation_score = row['evaluation_score']

    if completion_rate > 0.8 and evaluation_score > 80:
        return 2   # High
    elif completion_rate > 0.5 and evaluation_score > 50:
        return 1   # Medium
    else:
        return 0   # Low

df["performance_label"] = df.apply(assign_label, axis=1)

# Features & Target
X = df[[
    'attendance_percentage',
    'consistency_score',
    'learning_score',
    'tasks_assigned',
    'tasks_completed',
    'completion_rate',
    'project_score',
    'evaluation_score',
    'login_frequency',
    'active_days',
    'engagement_score',
    'communication_score',
    'discipline_score'
]]
y = df["performance_label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Before SMOTE:")
print(df["performance_label"].value_counts())


print("Class distribution:", Counter(y))

counts = Counter(y)
if min(counts.values()) > 1:
    smote = SMOTE(k_neighbors=1)
    X_res, y_res = smote.fit_resample(X_scaled, y)
else:
    print("⚠️ Skipping SMOTE because a class has only 1 sample. Add more data to the CSV!")
    X_res, y_res = X_scaled, y

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    objective='multi:softprob'
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler saved successfully ✅")
print("Model saved successfully ✅")
print("\nNew Class Distribution:")
print(pd.Series(y).value_counts())