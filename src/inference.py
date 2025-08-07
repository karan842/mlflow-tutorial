from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
import pandas as pd 
import mlflow 
import dagshub 
import mlflow.sklearn

dagshub.init(repo_owner="your-repo-owner", repo_name="your-repo-name", mlflow=True)
mlflow.set_tracking_uri("your-mlflow-tracking-uri")

# Load the produciton model
model_name = "tuned-breast-cancer"
model = mlflow.sklearn.load_model(f"models:/{model_name}/1")

# Load the breast cancer dataset
data = load_breast_cancer() 
pred_data = pd.DataFrame(data.data, columns=data.feature_names)
pred_data['target'] = data.target

pred_data = pred_data.drop("target", axis=1).head(1)

y_pred = model.predict(pred_data)

print(y_pred)
