import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(model, 'models/naive_bayes_model.pkl')
    return model, accuracy, y_pred

def train_bayesian(X_train, y_train, X_test, y_test):
    # Using Logistic Regression with probabilistic interpretation
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(model, 'models/bayesian_model.pkl')
    return model, accuracy, y_pred

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    joblib.dump(model, 'models/random_forest_model.pkl')
    return model, accuracy, y_pred

def compare_models(accuracies, output_dir='visualizations/'):
    # Ensure visualizations directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig(f'{output_dir}model_comparison.png')
    plt.close()

def main():
    print("Loading data...")
    # Load preprocessed data
    X_train_scaled = joblib.load('data/X_train_scaled.pkl')
    X_test_scaled = joblib.load('data/X_test_scaled.pkl')
    y_train = joblib.load('data/y_train.pkl')
    y_test = joblib.load('data/y_test.pkl')
    print("Data loaded successfully.")

    print("Training models...")
    # Train models
    nb_model, nb_accuracy, nb_pred = train_naive_bayes(X_train_scaled, y_train, X_test_scaled, y_test)
    bayes_model, bayes_accuracy, bayes_pred = train_bayesian(X_train_scaled, y_train, X_test_scaled, y_test)
    rf_model, rf_accuracy, rf_pred = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Print classification reports
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, nb_pred))
    print("Bayesian (Logistic Regression) Classification Report:")
    print(classification_report(y_test, bayes_pred))
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    # Compare models
    accuracies = {
        'Naive Bayes': nb_accuracy,
        'Bayesian': bayes_accuracy,
        'Random Forest': rf_accuracy
    }
    compare_models(accuracies)
    
    print("\nModel Accuracies:")
    for model, acc in accuracies.items():
        print(f"{model}: {acc:.4f}")

if __name__ == "__main__":
    main()