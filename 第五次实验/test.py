import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import linear_model

# 数据集
diabetes = datasets.load_diabetes()  # 载入数据

# 获取一个特征
diabetes_x_temp = diabetes.data[:, np.newaxis, 2]

X_train = diabetes_x_temp[:-20]  # 训练样本
X_test = diabetes_x_temp[-20:]  # 测试样本 后20行
y_train = diabetes.target[:-20]  # 训练标记
y_test = diabetes.target[-20:]  # 预测对比标记

# 回归训练及预测
clf = linear_model.LinearRegression()
clf.fit(X_train,y_train)  # 注: 训练数据集

# 系数 残差平法和 方差得分
print('Coefficients :\n', clf.coef_)
print("Residual sum of square: %.2f" % np.mean((clf.predict(X_test) - y_test) ** 2))
print("variance score: %.2f" % clf.score(X_test, y_test))

# 绘图
plt.title(u'LinearRegression Diabetes')  # 标题
plt.xlabel(u'Attributes')  # x轴坐标
plt.ylabel(u'Measure of disease')  # y轴坐标
# 点的准确位置
plt.scatter(X_test, y_test, color='black')
# 预测结果 直线表示
plt.plot(X_test, clf.predict(X_test), color='blue', linewidth=3)
plt.show()

# 进行预测
y_pred = clf.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差：", mse)

# 计算可决系数R2
r2 = r2_score(y_test, y_pred)
print("可决系数R2：", r2)

# 进行PCA分析
pca = PCA()
pca.fit(X_train)

# 获取可解释方差比和贡献度累计比
explained_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_var_ratio)

# 构建第一个DataFrame
df1 = pd.DataFrame({"属性名": diabetes.feature_names, "可解释方差比": explained_var_ratio})
df1 = df1.sort_values(by=["可解释方差比"], ascending=False)
print("可解释方差比DataFrame：")
print(df1)

# 构建第二个DataFrame
df2 = pd.DataFrame({"属性名": diabetes.feature_names, "贡献度累计比": cumulative_var_ratio})
df2 = df2.sort_values(by=["贡献度累计比"], ascending=False)
print("贡献度累计比DataFrame：")
print(df2)
