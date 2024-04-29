import os
import torch
import pandas as pd
import csv

def find_excel_files(folder_path, processed_files=None):
    if processed_files is None:
        processed_files = set()
    excel_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                file_path = os.path.join(root, file_name)
                if file_path not in processed_files:
                    processed_files.add(file_path)
                    excel_files.append(file_path)
        for dir_name in dirs:
            excel_files.extend(find_excel_files(os.path.join(root, dir_name), processed_files))
    return excel_files


main_folder_path = r'D:\银商杯大赛\大赛价格数据-2022'
excel_files_list = find_excel_files(main_folder_path)

#移除非正常文件
for i in excel_files_list:
    df = pd.read_excel(i, usecols=[8], nrows=249, dtype=float)
    column_data = df.values.flatten()
    if column_data.size != 249:
        os.remove(i)

#创建涨幅数据
csvfile = open('output_change.csv', 'w', encoding='utf_8')
csv_writer = csv.writer(csvfile)
cnt = 1
for i in excel_files_list:
    df = pd.read_excel(i, usecols=[8], nrows=249, dtype=float)
    column_data = df.values.flatten()
    csv_writer.writerow([f'loc{cnt}'])
    csv_writer.writerow([i])
    csv_writer.writerow([column_data])
    csv_writer.writerow([f'size{column_data.size}'])
    csv_writer.writerow('')
    cnt += 1
csvfile.close()

#创建涨幅数据
csvfile = open('output_change.csv', 'w', encoding='utf_8')
csv_writer = csv.writer(csvfile)
cnt = 1
for i in excel_files_list:
    df = pd.read_excel(i, usecols=[9], nrows=249, dtype=float)
    column_data = df.values.flatten()
    csv_writer.writerow([f'loc{cnt}'])
    csv_writer.writerow([i])
    csv_writer.writerow([column_data])
    csv_writer.writerow([f'size{column_data.size}'])
    csv_writer.writerow('')
    cnt += 1
csvfile.close()