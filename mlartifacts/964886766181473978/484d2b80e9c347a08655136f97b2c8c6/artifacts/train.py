import mlflow 
import mlflow.sklearn 
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
from mlflow.models.signature import infer_signature

# load wine dataset
wine = load_wine()
X = wine.data 
y = wine.target 
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the params for RF model
max_depth = 5 
n_estimators = 15

mlflow.set_experiment("YT-MLOps-Exp1")
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # log artifacts using mlflow
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    print(accuracy)
    
    # Set tags
    mlflow.set_tags({"version": "1.0", "author": "Karan", "project": "wine-classification"})
    
    # Log the model
    mlflow.sklearn.log_model(rf, 'random-forest-model')
    