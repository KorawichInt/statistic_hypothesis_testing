import pandas as pd
import os


def read_and_write_iris_data(source, destination):
    column_names = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width', 'class']
    data = pd.read_csv(source, delimiter=',', names=column_names)
    data.to_csv(destination, index=False)


def check_file_exists(file_path):
    return os.path.exists(file_path)


if __name__ == "__main__":
    # read iris.data and write to csv format with columns name
    source = 'assets\iris\iris.data'
    destination = 'csv_repository\iris_data.csv'

    # check if iris_data.csv is exists, if not, write it
    iris_csv_check = check_file_exists(destination)
    if iris_csv_check == "False":
        read_and_write_iris_data(source, destination)
