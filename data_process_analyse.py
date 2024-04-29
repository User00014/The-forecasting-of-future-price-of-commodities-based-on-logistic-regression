import os
import numpy as np
import torch
import pandas as pd
import csv
import math

# 输入文件夹名，遍历文件夹内的xls类型文件，输出所有xls文件名组成的列表
def find_excel_files(folder_path, processed_files=None):
    if processed_files is None:
        processed_files = set()
    excel_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 遍历到xls或xlsx文件名，若未记录则加入列表
            if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                file_path = os.path.join(root, file_name)
                if file_path not in processed_files:
                    processed_files.add(file_path)
                    excel_files.append(file_path)
        # 遍历子文件夹
        for dir_name in dirs:
            excel_files.extend(find_excel_files(os.path.join(root, dir_name), processed_files))
    return excel_files


main_folder_path = r'D:\银商杯大赛\大赛价格数据-2022'
excel_files_list = find_excel_files(main_folder_path)


main_folder_path = r'D:\银商杯大赛\大赛价格数据-2022'
excel_files_list = find_excel_files(main_folder_path)
csvfile = open('output.csv', 'w', encoding='utf_8')
csv_writer = csv.writer(csvfile)
cnt = 1

#将数据存储至output.csv，用于分析和测试代码，对之后的操作无影响，可以删除
for i in excel_files_list:
    df = pd.read_excel(i, usecols=[8], nrows=249, dtype=float)
    mean_value = df['平均价'].mean()
    df['平均价'].fillna(mean_value, inplace=True)

    column_data = df.values.flatten()
    csv_writer.writerow([f'loc{cnt}'])
    csv_writer.writerow([i])
    csv_writer.writerow([column_data])
    csv_writer.writerow([f'size{column_data.size}'])
    csv_writer.writerow('')
    cnt += 1
csvfile.close()


# 遍历列表中的xls或xlsx文件，取出第九列‘平均值’组合为249 * 686的张量X
def generate_tensor_from_excel_files(excel_files_list):
    all_data = []

    for i in excel_files_list:
        df = pd.read_excel(i, usecols=[8], nrows=249, dtype=float)

        mean_value = df['平均价'].mean()
        df['平均价'].fillna(mean_value, inplace=True)

        column_data = df.values.flatten()
        all_data.append(column_data)
    X = torch.tensor(all_data)
    return X


# 取出两个xls文件第九列组合成249 * 2的张量y
def get_y(file_path_1, file_path_2):
    all_data = []
    df = pd.read_excel(file_path_1, usecols=[0], nrows=249, dtype=float)
    column_data = df.values.flatten()
    all_data.append(column_data)
    df = pd.read_excel(file_path_2, usecols=[0], nrows=249, dtype=float)
    column_data = df.values.flatten()
    all_data.append(column_data)
    y = torch.tensor(all_data)
    return y


# 分别输出两个249 * 1的张量y1和y2
def get_y_seperate(file_path_1, file_path_2):
    data1 = []
    data2 = []
    df = pd.read_excel(file_path_1, usecols=[8], nrows=249, dtype=float)
    column_data = df.values.flatten()
    data1.append(column_data)
    df = pd.read_excel(file_path_2, usecols=[8], nrows=249, dtype=float)
    column_data = df.values.flatten()
    data2.append(column_data)
    y1 = torch.tensor(data1)
    y2 = torch.tensor(data2)
    return y1, y2


# 创建训练批次，输出feature和label的组合元组
def create_train_batch(input_data, batch, seq_len):
    feature_seq = []
    label_seq = []
    lens = len(input_data)
    for i in range (lens - batch):
        train_feature = input_data[i:i + batch - 1].float()
        train_label = input_data[i + batch].float()
        feature_seq.append(train_feature)
        label_seq.append(train_label)

    lstm_seqs = []
    lstm_labels = []
    for i in range(len(feature_seq)- batch - seq_len):
        tmp_seqs = []
        tmp_labels = []
        for j in range(seq_len):
            tmp_seqs.append(feature_seq[i + j])
        tmp_labels.append(label_seq[i + seq_len + batch - 1])
        tmp_seqs = torch.stack(tmp_seqs)
        tmp_labels = torch.stack(tmp_labels)
        lstm_seqs.append(tmp_seqs)
        lstm_labels.append(tmp_labels)

    return (lstm_seqs, lstm_labels)



# 标准化
def normalization_1(x):
    mean = torch.mean(x)
    square = torch.std(x)
    x = (x - mean) / (square + 1e-10).float()
    return x, mean, square


# 标准化还原
def map_to_score_1(pred, mean, std):
    gt = pred * std + mean
    return gt


def normalization(x):
    max_datum = torch.max(x)
    min_datum = torch.min(x)
    x = (x - min_datum) / (max_datum - min_datum + 1e-10)
    return x, max_datum, min_datum

def map_to_score(pred, max, min):
    gt = pred * (max - min) + min
    return gt


def create_inout_sequences(input_data, batch):
    inout_seq = []
    L = len(input_data)
    for i in range(L-batch):
        train_seq = input_data[i:i+batch]
        train_label = input_data[i+batch:i+batch+1]
        inout_seq.append((train_seq.T ,train_label))
    return inout_seq


def move(x, target, loss):
    original_std = np.std(x)
    # 计算扩大倍数
    scale_factor = math.sqrt(np.std(target) / original_std)
    # 对每个元素乘以扩大倍数
    scaled_data = [(i-np.mean(x)) * 2/scale_factor + np.mean(x) for i in x]
    x = [i + 6*loss for i in scaled_data]
    return x





