from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd
import os

def train_model(model_name, train, target, name):
    # Separate features (X) and target labels (y)
    X_train = train.drop(target, axis=1)
    y_train = train[target]

    # Select model based on the model_name argument
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=42)
    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "KNeighbors":
        model = KNeighborsClassifier()
    elif model_name == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose one from: 'RandomForest', 'XGBoost', 'LogisticRegression', 'SVM', 'KNeighbors', 'DecisionTree', 'GradientBoosting'.")

    model.fit(X_train, y_train)

    # Define the folder path and model name
    folder_name = f"{name}_ckpt"
    model_file_name = f"{model_name.lower()}_model.pkl"

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Save the model
    path = os.path.join(folder_name, model_file_name)
    joblib.dump(model, path)
    print(f"{model_name} model saved at '{path}'")

def evaluate_model(model_name, test, target, mode):
    # Load model
    model = joblib.load(f"{mode}_ckpt\\{model_name.lower()}_model.pkl")

    # Prepare test data
    X_test = test.drop(target, axis=1)
    y_test = test[target]
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, average='weighted')
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred, average='weighted')
    
    # Store metrics in a dictionary
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall
    }
    
    # Convert metrics to DataFrame and return it
    metrics_df = pd.DataFrame([metrics])

    return metrics_df