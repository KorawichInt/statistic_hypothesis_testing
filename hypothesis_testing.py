import pandas as pd
import os
import numpy as np


def check_file_exists(file_path):
    return os.path.exists(file_path)


def read_and_write_data(source_path, destination_path):
    data = pd.read_csv(source_path, delimiter=',')
    data.to_csv(destination_path, index=False)


def sampling_data(destination_path, destination_path2):
    data = pd.read_csv(destination_path, header=None)
    data_shuffled = data.sample(
        frac=1, random_state=42).reset_index(drop=True)
    group_sizes = [80, 38, 32]
    training_set = data_shuffled[:group_sizes[0]]
    validation_set = data_shuffled[group_sizes[0]                                   :group_sizes[0]+group_sizes[1]]
    testing_set = data_shuffled[group_sizes[0]+group_sizes[1]:]
    training_set.to_csv(destination_path2[0], index=False, header=False)
    validation_set.to_csv(destination_path2[1], index=False, header=False)
    testing_set.to_csv(destination_path2[2], index=False, header=False)


if __name__ == "__main__":
    # read iris.data and write to csv format with columns name
    source_path = r'assets\iris\iris.data'
    destination_path = r'csv_repository\iris_data.csv'
    destination_path2 = [r'csv_repository\training_set.csv',
                         r'csv_repository\validation_set.csv',
                         r'csv_repository\testing_set.csv']

    # check if iris_data.csv is not exists
    check_iris_csv = check_file_exists(destination_path)
    if not (check_iris_csv):
        print("1")
        read_and_write_data(source_path, destination_path)
    print("Already 1")

    # check if iris_data.csv is exists and training_set is not exists
    check_training_set = check_file_exists(destination_path2[0])
    if check_iris_csv and (check_training_set):
        print("2")
        sampling_data(destination_path, destination_path2)
    print("Already 2")
