import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_clean_data(filepath):
    print("--- Step 1: Loading and Cleaning Data ---")
    # Load dataset
    df = pd.read_csv(r'D:\Data Analytics Project\census-income.csv')
    
    # Rename target column for clarity
    df.rename(columns={'annual_income': 'income'}, inplace=True)
    
    # Handle missing values encoded as '?'
    df = df.replace("?", np.nan)
    print(f"Total missing values found after replacing '?': {df.isnull().sum().sum()}")
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Remove duplicate rows
    print(f"Duplicate rows found: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    
    return df

def remove_outliers(df):
    print("\n--- Step 2: Treating Outliers via IQR Method ---")
    # Columns identified for outlier treatment
    out_list = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
    
    initial_shape = df.shape[0]
    for col in out_list:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        LB = Q1 - 1.5 * IQR
        UB = Q3 + 1.5 * IQR
        
        # Filter dataframe within bounds
        df = df[(df[col] >= LB) & (df[col] <= UB)]
        
    final_shape = df.shape[0]
    print(f"Removed {initial_shape - final_shape} outlier rows.")
    return df

def encode_features(df):
    print("\n--- Step 3: Encoding Categorical Features ---")
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

def evaluate_models(X_train, X_test, y_train, y_test):
    print("\n--- Step 4: Model Training and Evaluation ---")
    results = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results['Logistic Regression'] = accuracy_score(y_test, lr_pred)
    
    # 2. Decision Tree Hyperparameter Tuning
    print("Tuning Decision Tree depths...")
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_dt_acc = 0
    best_depth = 1
    
    for d in depths:
        dt_model = DecisionTreeClassifier(max_depth=d, random_state=12)
        dt_model.fit(X_train, y_train)
        pred = dt_model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        if acc > best_dt_acc:
            best_dt_acc = acc
            best_depth = d
            
    results[f'Decision Tree (Max Depth={best_depth})'] = best_dt_acc
    
    # 3. Random Forest (Bug Fixed: Using rf_pred instead of y_pred)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, rf_pred)
    
    return results

if __name__ == "__main__":
    # Update this path if necessary
    data_path = 'D:\Data Analytics Project\census-income.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Please place your dataset at {data_path}")
    else:
        # Pipeline execution
        df = load_and_clean_data(data_path)
        df = remove_outliers(df)
        df = encode_features(df)
        
        # Features and Target selection
        X = df.iloc[:, :-1]
        y = df['income']
        
        # Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Evaluate Models
        model_accuracies = evaluate_models(X_train, X_test, y_train, y_test)
        
        # Print Final Comparison Report
        print("\n=============================================")
        print("         FINAL MODEL ACCURACY COMPARISON     ")
        print("=============================================")
        for model_name, accuracy in model_accuracies.items():
            print(f"{model_name:<30} : {accuracy * 100:.2f}%")
        print("=============================================")
        
        best_model = max(model_accuracies, key=model_accuracies.get)
        print(f"Winner Model based on Accuracy: {best_model}")