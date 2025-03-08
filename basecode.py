import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns

# Load dataset
df = pd.read_csv("student-scores-updated.csv")

# Encode categorical data
categorical_cols = ['gender', 'part_time_job', 'career_aspiration', 'extracurricular_activities']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and labels
features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
            'weekly_self_study_hours', 'career_aspiration', 'math_score', 'history_score', 
            'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']

# Calculate total score and risk status
df['total_score'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 
                        'biology_score', 'english_score', 'geography_score']].mean(axis=1)

# Adjusted Threshold for Risk Classification
df['risk_status'] = np.where((df['total_score'] < 60) | (df['absence_days'] > 5), 1, 0)

# Check the distribution of risk_status
print(df['risk_status'].value_counts())

X = df[features]
y = df['risk_status']

# Standardizing numerical features
scaler = StandardScaler()
numerical_cols = ['absence_days', 'weekly_self_study_hours', 'math_score', 'history_score', 
                  'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=class_weights[1])

# Train the XGBoost model
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Plot
plt.figure(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance in Student Risk Prediction")
plt.show()

# Risk Prediction with Expanded Intervention Strategies
def predict_risk():
    print("\nEnter student details for risk prediction:")
    input_data = {}

    for col in features:
        if col in ['part_time_job', 'extracurricular_activities']:
            user_input = input(f"Enter {col} (yes/no): ").strip().lower()
            if user_input not in ["yes", "no"]:
                print(f"Invalid input for {col}. Please enter 'yes' or 'no'.")
                return
            input_data[col] = 1 if user_input == "yes" else 0
        elif col == "career_aspiration":
            print(f"career_aspiration options: {list(label_encoders[col].classes_)}")
            user_input = input(f"Enter career_aspiration: ").strip()
            if user_input not in label_encoders[col].classes_:
                print(f"Invalid input for career_aspiration. Please enter one of the given options.")
                return
            input_data[col] = label_encoders[col].transform([user_input])[0]
        elif col in categorical_cols:
            options = list(label_encoders[col].classes_)
            print(f"{col} options: {options}")
            user_input = input(f"Enter {col}: ").strip()
            if user_input not in options:
                print(f"Invalid input for {col}. Please enter one of the given options.")
                return
            input_data[col] = label_encoders[col].transform([user_input])[0]
        else:
            try:
                input_data[col] = float(input(f"Enter {col}: "))
            except ValueError:
                print(f"Invalid input for {col}. Please enter a numerical value.")
                return

    # Convert input data to DataFrame and standardize numerical values
    input_df = pd.DataFrame([input_data])
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Risk Classification
    prediction = model.predict(input_df)[0]
    total_score = np.mean([input_data[col] for col in ['math_score', 'history_score', 'physics_score', 
                                                       'chemistry_score', 'biology_score', 'english_score', 
                                                       'geography_score']])
    risk_status = "At Risk" if (prediction == 1 or total_score < 60 or input_data['absence_days'] > 5) else "Not At Risk"
    
    print(f"\nPredicted Risk Status: {risk_status}")

    # Generate Intervention Report
    report = f"Student Risk Report:\n- Predicted Status: {risk_status}\n"
    if risk_status == "At Risk":
        interventions = []
        if total_score < 60:
            interventions.append("- Increase study hours and seek tutoring for weak subjects.")
            interventions.append("- Attend academic workshops for better understanding of key concepts.")
        if input_data['absence_days'] > 5:
            interventions.append("- Reduce absenteeism and maintain a regular study schedule.")
            interventions.append("- Consult a mentor or counselor to identify attendance issues.")
        if input_data['weekly_self_study_hours'] < 5:
            interventions.append("- Develop a structured study plan and set daily learning goals.")
        if input_data['extracurricular_activities'] == 0:
            interventions.append("- Engage in extracurricular activities to improve time management and social skills.")
        if input_data['career_aspiration'] == label_encoders['career_aspiration'].transform(["Unknown"])[0]:
            interventions.append("- Seek career counseling to identify interests and set clear academic goals.")

        report += "- Recommended Actions:\n" + "\n".join(interventions)

    with open("intervention_report.txt", "w") as file:
        file.write(report)

    print("\nIntervention report generated: intervention_report.txt")

# Run user input prediction
predict_risk()
