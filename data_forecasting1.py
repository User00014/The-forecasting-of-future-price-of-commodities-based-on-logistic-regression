import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process_analyse import *
import math
from torch.nn.utils import clip_grad_norm_
import csv

#导入数据
file_path_1 = "D:\银商杯大赛\大赛价格数据-2022\中游同行竞品\LLDPE\市场价格\LLDPE-市场价格-上海市场-上海赛科-0209AA(20220104-20221230).xls"
file_path_2 = "D:\银商杯大赛\大赛价格数据-2022\中游同行竞品\共聚PP粒\市场价格\共聚PP粒-市场价格-杭州市场-上海赛科-K4912(20220104-20221230).xls"

# 需要预测的序列 y（249*1）
Y1, Y2 = get_y_seperate(file_path_1, file_path_2)
Y1 = Y1.T

train_num = 200
Y1_train = Y1[:train_num - 1]
Y1_test = Y1[train_num:]
Y1_train, max_val, min_val = normalization(Y1_train)
Y1_train = Y1_train.to(dtype=torch.float32)

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_layer_size, output_size):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#         self.linear = nn.Sequential(
#             nn.Linear(hidden_layer_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_size)
#         )
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]

#根据已知条件确认参数
input_size = 1
hidden_layer_size = 8
output_size = 1

#调整训练中涉及的超参数
learning_rate = 0.0005
epochs = 12
batch = 135

model = nn.Sequential(
            nn.Linear(batch, 16),
            nn.ReLU(),
            nn.Dropout(p=0.99),
            # nn.Linear(8, 16),
            # nn.ReLU(),
            # nn.Dropout(p=0.99),
            nn.Linear(16, output_size)
        )


train_inout_seq1 = create_inout_sequences(Y1_train, batch)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()


for epoch in range(epochs+1):
    for features, labels in train_inout_seq1:
        optimizer.zero_grad()
        y_hat = model(features)
        loss = loss_function(y_hat, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

    print(f'Epoch [{epoch}/{epochs}], Training Loss: {loss:.8f}')
print('end')

future_pred = 61 + len(Y1_test)
data = Y1_train[-batch:].tolist()
data = [sublist for items in data for sublist in items ]
model.eval()

for i in range(future_pred):
    pre_features = torch.FloatTensor(data[-batch:])
    with torch.no_grad():
        prediction = model(pre_features).item()
        data.append(prediction)

for i in range(len(data)):
    data[i] = data[i] * (max_val.item() - min_val.item()) + min_val.item()

Y1_test = Y1_test.numpy()
test_loss = 0
for i in range(len(Y1_test)):
    test_loss += abs(Y1_test[i] - data[batch + i])
test_loss = test_loss/len(Y1_test)
print(test_loss)

data = move(data, Y1_test, test_loss)

with open("file1.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)

    x = np.arange(future_pred)
    plt.plot(x, data[batch:], color='blue')
    plt.plot(np.arange(len(Y1_test)), Y1_test, color='red')
    plt.show()

    csv_writer.writerow(data[batch+len(Y1_test):])
