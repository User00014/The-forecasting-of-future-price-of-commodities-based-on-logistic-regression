import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalization(x):
    max_datum = torch.max(x).item()
    min_datum = torch.min(x).item()
    x = (x - min_datum) / (max_datum - min_datum + 1e-10)
    return x, max_datum, min_datum


def read_second_column_and_convert_to_tensor(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    # 获取第二列数据，去除第一行，取前 366 行数据
    column_data = df.iloc[1:367, 1].values
    # 将数据转换为张量
    tensor_data = torch.tensor(column_data, dtype=torch.float32)
    ans, max, min = normalization(tensor_data)
    return ans, max, min


# 提取相应的文件内容
file_path = r"C:\Users\13488\Desktop\上海地铁客流量.xlsx"
data, max, min = read_second_column_and_convert_to_tensor(file_path)
batch_size = 50

def split_dataset(data, train_ratio=0.8):
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    train_data, test_data = data[:train_size], data[train_size:]
    return train_data, test_data

def create_inout_sequences(input_data, batch = batch_size):
    inout_seq = []
    L = len(input_data)
    for i in range(L-batch):
        train_seq = input_data[i:i+batch]
        train_label = input_data[i+batch:i+batch+1]
        inout_seq.append((train_seq.T ,train_label))
    return inout_seq


def map_to_score(pred, max, min):
    gt = pred * (max - min) + min
    return gt

train_data, test_data = split_dataset(data)
train_inout = create_inout_sequences(train_data)
test_inout = create_inout_sequences(test_data)


# 训练模型
learning_rate = 0.001
model = nn.Sequential(
            nn.Linear(batch_size, 16),
            nn.ReLU(),
            nn.Dropout(p=0.99),
            nn.Linear(16, 1)
        )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()
epochs = 10

for epoch in range(epochs+1):
    for features, labels in train_inout:
        optimizer.zero_grad()
        y_hat = model(features)
        loss = loss_function(y_hat, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch}/{epochs}], Training Loss: {loss:.8f}')
print('end')


# 根据训练好的模型进行预测
future_pred = 110 + len(test_data)
data_pre = train_data[-batch_size:].tolist()

model.eval()
for i in range(future_pred):
    pre_features = torch.FloatTensor(data[-batch_size:])
    with torch.no_grad():
        prediction = model(pre_features).item()
        data_pre.append(prediction)

for i in range(len(data_pre)):
    data_pre[i] = data_pre[i] * (max - min) + min
for i in range(len(test_data)):
    test_data[i] = test_data[i] * (max - min) + min

x = np.arange(future_pred)
plt.plot(x, data_pre[batch_size:], color='blue')
plt.plot(np.arange(len(test_data)), test_data, color='red')
plt.show()

def save_to_excel(data, file_path):
    df = pd.DataFrame(data, columns=["Predictions"])
    df.to_excel(file_path, index=False)
    print(f"Data saved to {file_path}.")

# 示例用法：将predictions保存到Excel文件中
file_path = file_path = r"C:\Users\13488\Desktop\output.xlsx"
save_to_excel(data_pre, file_path)

print(data_pre)
