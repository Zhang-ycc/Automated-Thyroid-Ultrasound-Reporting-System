import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
import pickle

# 1. 数据准备
# 假设你已经准备好了训练数据 X_train（特征）和 y_train（对应的灰度阈值标签）
# X_train 的形状为 (样本数量, 特征数量)，y_train 的形状为 (样本数量,)

# 2. 数据划分（示例中简单将所有数据用于训练，可以自行划分训练集和测试集）
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# 3. 模型训练
# 创建 SVM 模型对象，设置相关参数（如核函数、正则化参数等）
model = svm.SVR(kernel='linear', C=0.1)
# 使用训练集的特征和标签进行训练
model.fit(X_train, y_train)

# 4. 模型评估
# 在训练集上进行预测
y_train_pred = model.predict(X_train)
train_error = mean_squared_error(y_train, y_train_pred)
print(f"训练集均方误差：{train_error}")

# 在测试集上进行预测
y_test_pred = model.predict(X_test)
print(y_test_pred)
test_error = mean_squared_error(y_test, y_test_pred)
print(f"测试集均方误差：{test_error}")

# 保存模型
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


# # 5. 模型应用
# # 当模型达到满意的性能后，可以使用该模型来预测新图像的阈值
# new_image = ...
# new_image_features = ...
# threshold_prediction = model.predict(new_image_features)
# print(f"预测的阈值：{threshold_prediction}")
