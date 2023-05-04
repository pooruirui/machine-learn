# -*- coding = utf-8 -*-
# @Time : 2023/4/7 0:47
# @Author : 彭睿
# @File : best_parameters.py.py
# @Software : PyCharm
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import matplotlib.pyplot as plt

def run_model(X_train, X_test, y_train, y_test, model_type, dataset):
    if dataset == 'iris' or dataset == 'breast_cancer':
        if model_type == 'cart':
            clf = DecisionTreeClassifier(random_state=42)
        elif model_type == 'bagging':
            clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), random_state=42)
        elif model_type == 'rf':
            clf = RandomForestClassifier(random_state=42)
        elif model_type == 'gbdt':
            clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

iris = load_iris()
breast_cancer = load_breast_cancer()
boston = load_boston()

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
bc_X_train, bc_X_test, bc_y_train, bc_y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=0)
boston_X_train, boston_X_test, boston_y_train, boston_y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)

iris_cart = run_model(iris_X_train, iris_X_test, iris_y_train, iris_y_test, 'cart', 'iris')
iris_bagging = run_model(iris_X_train, iris_X_test, iris_y_train, iris_y_test, 'bagging', 'iris')
iris_rf = run_model(iris_X_train, iris_X_test, iris_y_train, iris_y_test, 'rf', 'iris')
iris_gbdt = run_model(iris_X_train, iris_X_test, iris_y_train, iris_y_test, 'gbdt', 'iris')

#breast分类
clf = DecisionTreeClassifier(random_state=42,criterion='entropy',splitter='random',min_samples_split=21)
clf.fit(bc_X_train,bc_y_train)
y_pred = clf.predict(bc_X_test)
bc_cart = accuracy_score(bc_y_test, y_pred)

clf = BaggingClassifier(DecisionTreeClassifier(random_state=42,criterion='entropy',splitter='random',min_samples_split=21)
                     ,random_state=42,n_estimators=10,max_samples=0.49)
clf.fit(bc_X_train,bc_y_train)
y_pred = clf.predict(bc_X_test)
bc_bagging = accuracy_score(bc_y_test, y_pred)

clf = RandomForestClassifier(random_state=42,n_estimators=100)
clf.fit(bc_X_train,bc_y_train)
y_pred = clf.predict(bc_X_test)
bc_rf = accuracy_score(bc_y_test, y_pred)

clf = GradientBoostingClassifier(random_state=42,learning_rate=0.09,n_estimators=100,max_depth=3,min_samples_split=2,min_samples_leaf=32)
clf.fit(bc_X_train,bc_y_train)
y_pred = clf.predict(bc_X_test)
bc_gbdt = accuracy_score(bc_y_test, y_pred)


#boston回归
clf = DecisionTreeRegressor(random_state=42,criterion='mse',splitter='best',min_samples_split=25)
clf.fit(boston_X_train,boston_y_train)
y_pred = clf.predict(boston_X_test)
boston_cart = r2_score(boston_y_test, y_pred)

clf = BaggingRegressor(DecisionTreeRegressor(random_state=42,criterion='mse',splitter='best',min_samples_split=25),
                        random_state=42,n_estimators=32,max_samples=0.97,bootstrap=True)
clf.fit(boston_X_train,boston_y_train)
y_pred = clf.predict(boston_X_test)
boston_bagging = r2_score(boston_y_test, y_pred)

clf = RandomForestRegressor(random_state=42,n_estimators=185,min_samples_split=10,min_samples_leaf=1,n_jobs=4)
clf.fit(boston_X_train,boston_y_train)
y_pred = clf.predict(boston_X_test)
boston_rf = r2_score(boston_y_test, y_pred)

clf = GradientBoostingRegressor(random_state=42,learning_rate=0.25,n_estimators=435,min_samples_split=14,min_samples_leaf=1,
                                subsample=1)
clf.fit(boston_X_train,boston_y_train)
y_pred = clf.predict(boston_X_test)
boston_gbdt = r2_score(boston_y_test, y_pred)

x = np.array(['cart','bagging','rf','gdbt'])

iris_y = np.array([iris_cart, iris_bagging, iris_rf, iris_gbdt])
bc_y = np.array([bc_cart, bc_bagging, bc_rf, bc_gbdt])
boston_y = np.array([boston_cart, boston_bagging, boston_rf, boston_gbdt])

fig, ax = plt.subplots()
ax.scatter(x[:4], iris_y, marker='s',label='iris')
ax.scatter(x[:4], bc_y, marker='o',label='bc')
ax.scatter(x[:4], boston_y, marker='^',label='boston')
ax.plot(x[:4], iris_y, '-')
ax.plot(x[:4], bc_y, '-')
ax.plot(x[:4], boston_y, '-')
ax.scatter(x[:4], iris_y, marker='s', s=150, color='white', edgecolors='black')
ax.scatter(x[:4], bc_y, marker='o', s=150, color='white', edgecolors='black')
ax.scatter(x[:4], boston_y, marker='^', s=150, color='white', edgecolors='black')
ax.set_xlabel('Algorithm')
plt.legend()
ax.set_ylabel('Accuracy or R^2')
ax.set_title('Comparison of decision tree algorithms')
plt.show()

print('iris数据集的准确四个算法训练的准确率')
print(iris_y)
print('breast_cancer数据集的准确四个算法训练的准确率')
print(bc_y)
print('boston数据集的准确四个算法训练的准确率')
print(boston_y)