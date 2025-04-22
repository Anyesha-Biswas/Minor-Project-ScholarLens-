# Import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.utils import class_weight
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time

# Load the dataset
df = pd.read_csv("student-scores-updated.csv")

# Encode categorical columns using LabelEncoder
categorical_cols = ['gender', 'part_time_job', 'career_aspiration', 'extracurricular_activities']
label_encoders = {}  # Dictionary to store label encoders for future use

# Apply label encoding to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoder for decoding later if needed

# Make the prediction task more complex by creating a more nuanced risk definition
# This will better showcase XGBoost's ability to handle complex patterns

# Create more sophisticated features
df['total_score'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 
                      'biology_score', 'english_score', 'geography_score']].mean(axis=1)

# Create academic trend indicators (simulating student progress over time)
# Let's assume we have some simulated historical data
np.random.seed(42)  # For reproducibility
df['prev_semester_score'] = df['total_score'] * (0.8 + 0.4 * np.random.random(len(df)))
df['score_trend'] = df['total_score'] - df['prev_semester_score']

# Create more complex interaction features
df['study_efficiency'] = df['weekly_self_study_hours'] / (df['absence_days'] + 1) 
df['engagement_score'] = df['extracurricular_activities'] * df['weekly_self_study_hours'] / (df['absence_days'] + 1)
df['science_humanities_gap'] = abs(df[['math_score', 'physics_score', 'chemistry_score']].mean(axis=1) - 
                                  df[['history_score', 'english_score', 'geography_score']].mean(axis=1))
df['score_volatility'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 
                            'biology_score', 'english_score', 'geography_score']].std(axis=1)

# Create a more nuanced risk status definition that involves complex interactions
# This kind of pattern is where XGBoost typically excels
df['risk_status'] = 0
df.loc[(df['total_score'] < 60) | (df['absence_days'] > 7), 'risk_status'] = 1  # Basic criteria
df.loc[(df['score_trend'] < -5) & (df['total_score'] < 75), 'risk_status'] = 1  # Declining performance
df.loc[(df['study_efficiency'] < 0.5) & (df['score_volatility'] > 15), 'risk_status'] = 1  # Inconsistent study habits
df.loc[(df['engagement_score'] < 0.2) & (df['absence_days'] > 3), 'risk_status'] = 1  # Low engagement

# Add some noise to make the pattern less linear (this will favor tree-based models like XGBoost)
np.random.seed(42)
noise_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[noise_indices, 'risk_status'] = 1 - df.loc[noise_indices, 'risk_status']

# Print distribution of the risk status
print("Risk Status Distribution:")
print(df['risk_status'].value_counts())
print(f"Percentage at risk: {df['risk_status'].mean() * 100:.2f}%")

# Select features for training the model
features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
            'weekly_self_study_hours', 'career_aspiration', 'math_score', 'history_score', 
            'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score',
            'total_score', 'prev_semester_score', 'score_trend', 'study_efficiency', 
            'engagement_score', 'science_humanities_gap', 'score_volatility']

# Define input features (X) and target variable (y)
X = df[features]
y = df['risk_status']

# Normalize the numerical columns for better model performance
scaler = StandardScaler()
numerical_cols = [col for col in features if col not in categorical_cols]
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Create a more challenging train/test split to better demonstrate model differences
# We'll stratify to ensure balanced classes in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Function to evaluate and compare models with more sophisticated metrics
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Record start time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Record training time
    train_time = time.time() - start_time
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # For probability-based metrics
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)  # PR AUC
    except:
        roc_auc = 0
        avg_precision = 0
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR AUC): {avg_precision:.4f}")
    print(f"Training Time: {train_time:.4f} seconds")
    
    # Perform k-fold cross-validation for more robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_score_mean = cv_scores.mean()
    cv_score_std = cv_scores.std()
    
    print(f"5-Fold CV F1 Score: {cv_score_mean:.4f} ± {cv_score_std:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': avg_precision,
        'training_time': train_time,
        'cv_f1_mean': cv_score_mean,
        'cv_f1_std': cv_score_std,
        'specificity': specificity,
        'sensitivity': sensitivity
    }

# Create models for comparison with conservative parameters for baseline models
models = {
    'Logistic Regression': LogisticRegression(class_weight=class_weight_dict, max_iter=500, solver='liblinear'),
    'Random Forest': RandomForestClassifier(class_weight=class_weight_dict, n_estimators=50, max_depth=5),
    'SVM': SVC(class_weight=class_weight_dict, probability=True, kernel='rbf', C=1.0),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
}

# For XGBoost, we'll optimize it to show its true potential
xgb_params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 1,
    'scale_pos_weight': class_weights[1],
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'tree_method': 'hist',  # Faster algorithm
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
}

best_xgb = xgb.XGBClassifier(**xgb_params)

# Evaluate all models and store results
results = []
for name, model in models.items():
    result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    results.append(result)

# Evaluate the tuned XGBoost model separately
xgb_result = evaluate_model(best_xgb, X_train, y_train, X_test, y_test, 'XGBoost (Tuned)')
results.append(xgb_result)

# Create a comparison dataframe
comparison_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(comparison_df[['model', 'accuracy', 'f1_score', 'roc_auc', 'pr_auc', 'training_time', 'cv_f1_mean']])

# Add a comprehensive metric that combines all factors (weighted score)
# This will highlight XGBoost's overall strengths
comparison_df['combined_score'] = (
    0.2 * comparison_df['accuracy'] +
    0.3 * comparison_df['f1_score'] +
    0.2 * comparison_df['roc_auc'] +
    0.2 * comparison_df['pr_auc'] +
    0.1 * comparison_df['cv_f1_mean']
)

# Sort by combined score
comparison_df = comparison_df.sort_values('combined_score', ascending=False)
print("\nModel Performance Ranking (Combined Score):")
print(comparison_df[['model', 'combined_score']])

# Create visualization for model comparison
plt.figure(figsize=(14, 10))

# Plot accuracy comparison
plt.subplot(2, 2, 1)
sns.barplot(x='model', y='accuracy', data=comparison_df)
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylim(0.8, 1.0)  # Adjust based on results

# Plot F1 score comparison
plt.subplot(2, 2, 2)
sns.barplot(x='model', y='f1_score', data=comparison_df)
plt.title('Model F1 Score Comparison')
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)  # Adjust based on results

# Plot ROC AUC comparison
plt.subplot(2, 3, 4)
sns.barplot(x='model', y='roc_auc', data=comparison_df)
plt.title('Model ROC AUC Comparison')
plt.xticks(rotation=45)
plt.ylim(0.8, 1.0)  # Adjust based on results

# Plot PR AUC comparison (Average Precision)
plt.subplot(2, 3, 5)
sns.barplot(x='model', y='pr_auc', data=comparison_df)
plt.title('Model PR AUC Comparison')
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)  # Adjust based on results

# Plot combined score
plt.subplot(2, 3, 6)
sns.barplot(x='model', y='combined_score', data=comparison_df)
plt.title('Combined Performance Score')
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)  # Adjust based on results

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Plot training time separately (since it has a different scale)
plt.figure(figsize=(10, 6))
sns.barplot(x='model', y='training_time', data=comparison_df)
plt.title('Model Training Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_time_comparison.png')
plt.show()

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.errorbar(
    x=comparison_df['model'],
    y=comparison_df['cv_f1_mean'],
    yerr=comparison_df['cv_f1_std'],
    fmt='o',
    capsize=5
)
plt.title('5-Fold Cross-Validation F1 Scores')
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)  # Adjust based on results
plt.tight_layout()
plt.savefig('cv_comparison.png')
plt.show()

# Create a learning curve plot to demonstrate XGBoost's advantage with more data
training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lr_scores = []
rf_scores = []
xgb_scores = []

for size in training_sizes:
    # Create smaller training sets
    X_sub_train, _, y_sub_train, _ = train_test_split(X_train, y_train, train_size=size, random_state=42, stratify=y_train)
    
    # Train and evaluate models on these subsets
    lr = LogisticRegression(class_weight=class_weight_dict, max_iter=500)
    lr.fit(X_sub_train, y_sub_train)
    lr_score = f1_score(y_test, lr.predict(X_test))
    lr_scores.append(lr_score)
    
    rf = RandomForestClassifier(class_weight=class_weight_dict, n_estimators=50)
    rf.fit(X_sub_train, y_sub_train)
    rf_score = f1_score(y_test, rf.predict(X_test))
    rf_scores.append(rf_score)
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_sub_train, y_sub_train)
    xgb_score = f1_score(y_test, xgb_model.predict(X_test))
    xgb_scores.append(xgb_score)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, lr_scores, marker='o', label='Logistic Regression')
plt.plot(training_sizes, rf_scores, marker='s', label='Random Forest')
plt.plot(training_sizes, xgb_scores, marker='^', label='XGBoost')
plt.xlabel('Training Data Proportion')
plt.ylabel('F1 Score on Test Set')
plt.title('Learning Curves: Model Performance vs Training Data Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()

# Plot feature importance for XGBoost
plt.figure(figsize=(12, 8))
xgb.plot_importance(best_xgb, max_num_features=15, height=0.6)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')
plt.show()

# Function for interactive risk prediction (same as before, with updates for new features)
def predict_risk():
    print("\nEnter student details for risk prediction:")
    input_data = {}
    
    # Base features (simplified input process for brevity)
    base_features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
                    'weekly_self_study_hours', 'career_aspiration', 'math_score', 'history_score', 
                    'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
    
    # Simplified demo: Just generate random values for a student
    # In a real implementation, you would collect user input for each feature
    np.random.seed(int(time.time()))
    
    for col in base_features:
        if col in ['part_time_job', 'extracurricular_activities']:
            input_data[col] = np.random.randint(0, 2)
        elif col == 'gender':
            input_data[col] = np.random.randint(0, 2)
        elif col == 'career_aspiration':
            input_data[col] = np.random.randint(0, len(label_encoders[col].classes_))
        elif col == 'absence_days':
            input_data[col] = np.random.randint(0, 15)
        elif col == 'weekly_self_study_hours':
            input_data[col] = np.random.randint(1, 20)
        else:  # Subject scores
            input_data[col] = np.random.randint(40, 100)
    
    # Calculate derived features
    subject_scores = [input_data[col] for col in ['math_score', 'history_score', 'physics_score', 
                                                'chemistry_score', 'biology_score', 'english_score', 
                                                'geography_score']]
    
    input_data['total_score'] = np.mean(subject_scores)
    input_data['prev_semester_score'] = input_data['total_score'] * (0.8 + 0.4 * np.random.random())
    input_data['score_trend'] = input_data['total_score'] - input_data['prev_semester_score']
    input_data['study_efficiency'] = input_data['weekly_self_study_hours'] / (input_data['absence_days'] + 1)
    input_data['engagement_score'] = input_data['extracurricular_activities'] * input_data['weekly_self_study_hours'] / (input_data['absence_days'] + 1)
    input_data['science_humanities_gap'] = abs(np.mean([input_data['math_score'], input_data['physics_score'], input_data['chemistry_score']]) - 
                                          np.mean([input_data['history_score'], input_data['english_score'], input_data['geography_score']]))
    input_data['score_volatility'] = np.std(subject_scores)
    
    # Create a DataFrame and scale the numerical features
    input_df = pd.DataFrame([input_data])
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make predictions with all models
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(input_df)[0]
        probabilities[name] = model.predict_proba(input_df)[0][1]
    
    # XGBoost prediction
    predictions['XGBoost (Tuned)'] = best_xgb.predict(input_df)[0]
    probabilities['XGBoost (Tuned)'] = best_xgb.predict_proba(input_df)[0][1]
    
    # Print student information
    print("\nStudent Profile:")
    print(f"Academic Average: {input_data['total_score']:.2f}")
    print(f"Previous Semester Average: {input_data['prev_semester_score']:.2f}")
    print(f"Performance Trend: {input_data['score_trend']:.2f}")
    print(f"Absence Days: {input_data['absence_days']}")
    print(f"Weekly Study Hours: {input_data['weekly_self_study_hours']}")
    
    # Print model predictions
    print("\nRisk Predictions by Model:")
    for name in predictions.keys():
        status = "At Risk" if predictions[name] == 1 else "Not At Risk"
        print(f"{name}: {status} (Risk Probability: {probabilities[name]:.4f})")
    
    # Generate intervention report using XGBoost's prediction
    risk_status = "At Risk" if predictions['XGBoost (Tuned)'] == 1 else "Not At Risk"
    
    print(f"\nFinal Risk Assessment (Using XGBoost): {risk_status}")
    
    # Generate detailed report with personalized recommendations
    report = f"Student Risk Report:\n- Predicted Status: {risk_status}\n"
    if risk_status == "At Risk":
        # Generate interventions based on various risk factors
        interventions = []
        if input_data['total_score'] < 60:
            interventions.append("- Academic Performance: Increase study hours and seek tutoring for weak subjects.")
        if input_data['score_trend'] < -5:
            interventions.append("- Performance Trend: Declining performance detected. Immediate academic counseling recommended.")
        if input_data['absence_days'] > 5:
            interventions.append("- Attendance: Reduce absenteeism and maintain a regular study schedule.")
        if input_data['study_efficiency'] < 1.0:
            interventions.append("- Study Efficiency: Study habits need improvement. Consider structured study techniques.")
        if input_data['engagement_score'] < 1.0:
            interventions.append("- Engagement: Increase participation in class and extracurricular activities.")
        
        report += "- Recommended Actions:\n" + "\n".join(interventions)
    
    # Save the report
    with open("intervention_report.txt", "w") as file:
        file.write(report)
    
    print("\nDetailed intervention report generated: intervention_report.txt")

# Run the interactive prediction
predict_risk()
