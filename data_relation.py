import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from data_process_analyse import *
import openpyxl
from torch import nn as nn

num_examples = 686  # 样本数量
num_points = 249  # 每组数量

#导入数据
xdata_folder_path = 'D:\银商杯大赛\大赛价格数据-2022'
excel_files_list_x = find_excel_files(xdata_folder_path)
file_path_1 = "D:\银商杯大赛\已处理数据\LLDPE-市场价格-上海市场.xlsx"
file_path_2 = "D:\银商杯大赛\已处理数据\共聚PP粒-市场价格-杭州市场.xls"

# 输入条件的自变量和因变量x（249*686） y（249*2）
X = generate_tensor_from_excel_files(excel_files_list_x).T
X,_,_ = normalization(X)
Y = get_y(file_path_1, file_path_2).T
Y, mean, std = normalization(Y)

# 将数据分成训练数据和测试数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = nn.Sequential(nn.Linear(num_examples,num_examples//2).float(),
        nn.Dropout(p=0.05),
    nn.Linear(num_examples//2,2).float())

# 定义优化器和学习率
learning_rate = 0.005
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    predictions = model(X_train.float())
    # 计算训练损失
    train_loss = torch.mean((predictions - Y_train) ** 2)
    # 反向传播和优化
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss.item():.4f}')

#在测试数据上评估模型
with torch.no_grad():
   test_predictions =map_to_score(model(X_test.float()), mean, std)
   Y_test = map_to_score(Y_test, mean, std)
   test_loss = torch.mean(abs(test_predictions - Y_test))
   for name, param in model.named_parameters():
       print(name, param.shape, param)

print(f'Test Loss: {test_loss.item():.4f}')




