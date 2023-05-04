# -*- coding = utf-8 -*-
# @Time : 2023/4/11 21:01
# @Author : 彭睿
# @File : test.py
# @Software : PyCharm
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# 读取数据
df = pd.read_csv('train.csv')

# 数据预处理
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(method='ffill')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Sex'] = preprocessing.LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = preprocessing.LabelEncoder().fit_transform(df['Embarked'])

# 划分训练集和测试集
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 随机森林
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)
rfc_accuracy = accuracy_score(y_test, rfc_y_pred)
rfc_precision = precision_score(y_test, rfc_y_pred)
rfc_recall = recall_score(y_test, rfc_y_pred)
rfc_fscore = f1_score(y_test, rfc_y_pred)

# Bagging提升
bc = BaggingClassifier(KNeighborsClassifier(), n_estimators=10, max_samples=0.5, max_features=0.5)
bc.fit(X_train, y_train)
bc_y_pred = bc.predict(X_test)
bc_accuracy = accuracy_score(y_test, bc_y_pred)
bc_precision = precision_score(y_test, bc_y_pred)
bc_recall = recall_score(y_test, bc_y_pred)
bc_fscore = f1_score(y_test, bc_y_pred)

# 逻辑回归
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
lr_fscore = f1_score(y_test, lr_y_pred)

# Naive bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_y_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_precision = precision_score(y_test, nb_y_pred)
nb_recall = recall_score(y_test, nb_y_pred)
nb_fscore = f1_score(y_test, nb_y_pred)

# KNN算法
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
knn_precision = precision_score(y_test, knn_y_pred)
knn_recall = recall_score(y_test, knn_y_pred)
knn_fscore = f1_score(y_test, knn_y_pred)

# 存入dataframe
data = {'Algorithm': ['RandomForest', 'Bagging', 'LogisticRegression', 'NaiveBayes', 'KNN'],
        'Accuracy': [rfc_accuracy, bc_accuracy, lr_accuracy, nb_accuracy, knn_accuracy],
'Precision': [rfc_precision, bc_precision, lr_precision, nb_precision, knn_precision],
'Recall': [rfc_recall, bc_recall, lr_recall, nb_recall, knn_recall],
'F-score': [rfc_fscore, bc_fscore, lr_fscore, nb_fscore, knn_fscore]}
df_result = pd.DataFrame(data)
df_result.set_index('Algorithm', inplace=True)

print(df_result)

# 读取测试数据
df_test = pd.read_csv('test.csv')

# 对测试集进行预处理
df_test['Age'] = df_test['Age'].fillna(df['Age'].mean())
df_test['Embarked'] = df_test['Embarked'].fillna(method='ffill')
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_test['Sex'] = preprocessing.LabelEncoder().fit_transform(df_test['Sex'])
df_test['Embarked'] = preprocessing.LabelEncoder().fit_transform(df_test['Embarked'])

# 对测试集进行标准化
df_test_scaled = preprocessing.StandardScaler().fit_transform(df_test)

# 对测试集进行预测
rfc_test_pred = rfc.predict(df_test)
bc_test_pred = bc.predict(df_test)
lr_test_pred = lr.predict(df_test)
nb_test_pred = nb.predict(df_test)
knn_test_pred = knn.predict(df_test)

# 将预测结果存储在一个数据帧中
data = {'Algorithm': ['RandomForest', 'Bagging', 'LogisticRegression', 'NaiveBayes', 'KNN'],
        'Test_Predictions': [rfc_test_pred, bc_test_pred, lr_test_pred, nb_test_pred, knn_test_pred]}
df_test_pred = pd.DataFrame(data)
df_test_pred.set_index('Algorithm', inplace=True)

print(df_test_pred)