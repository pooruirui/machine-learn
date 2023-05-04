from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import matplotlib.pyplot as plt

def run_model(X_train, X_test, y_train, y_test, model_type):
    models = {
        'cart': DecisionTreeClassifier(random_state=42),
        'bagging': BaggingClassifier(DecisionTreeClassifier(random_state=42), random_state=42),
        'rf': RandomForestClassifier(random_state=42),
        'gbdt': GradientBoostingClassifier(random_state=42),
        'cart_reg': DecisionTreeRegressor(random_state=42),
        'bagging_reg': BaggingRegressor(DecisionTreeRegressor(random_state=42), random_state=42),
        'rf_reg': RandomForestRegressor(random_state=42),
        'gbdt_reg': GradientBoostingRegressor(random_state=42),
    }
    clf = models[model_type]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if 'Regressor' in str(type(clf)):
        score = r2_score(y_test, y_pred)
    else:
        score = accuracy_score(y_test, y_pred)
    return score

iris = load_iris()
breast_cancer = load_breast_cancer()
boston = load_boston()

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
bc_X_train, bc_X_test, bc_y_train, bc_y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=0)
boston_X_train, boston_X_test, boston_y_train, boston_y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)

iris_scores = [run_model(iris_X_train, iris_X_test, iris_y_train, iris_y_test, model) for model in ['cart', 'bagging', 'rf', 'gbdt']]
bc_scores = [run_model(bc_X_train, bc_X_test, bc_y_train, bc_y_test, model) for model in ['cart', 'bagging', 'rf', 'gbdt']]
boston_scores = [run_model(boston_X_train, boston_X_test, boston_y_train, boston_y_test, model) for model in ['cart_reg', 'bagging_reg', 'rf_reg', 'gbdt_reg']]

x = np.array(['cart', 'bagging', 'rf', 'gbdt'])

fig, ax = plt.subplots()
ax.scatter(x, iris_scores, marker='s', label='iris')
ax.scatter(x, bc_scores, marker='o', label='bc')
ax.scatter(x, boston_scores, marker='^', label='boston')
ax.plot(x, iris_scores, '-')
ax.plot(x, bc_scores, '-')
ax.plot(x, boston_scores, '-')
ax.scatter(x, iris_scores, marker='s', s=150, color='white', edgecolors='black')
ax.scatter(x, bc_scores, marker='o', s=150, color='white', edgecolors='black')
ax.scatter(x, boston_scores, marker='^', s=150, color='white', edgecolors='black')
ax.set_xlabel('Algorithm')
plt.legend()
ax.set_ylabel('Accuracy or R^2')
ax.set_title('Comparison of decision tree algorithms')
plt.show()
