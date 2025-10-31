import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
import joblib
import yaml
import mlflow
import mlflow.sklearn

# Read parameters
params = yaml.safe_load(open("params.yaml"))['train']

df = pd.read_csv('data/processed/processed_data.csv')

# Target column
X = df.drop('line_item_value', axis=1)
y = df['line_item_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['test_size'], random_state=params['random_state']
)

mlflow.set_experiment("Consignment Pricing")

with mlflow.start_run():
    if params['model_type'] == 'random_forest':
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
    elif params['model_type'] == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = LassoCV(cv=5)

    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, 'models/model.pkl')
