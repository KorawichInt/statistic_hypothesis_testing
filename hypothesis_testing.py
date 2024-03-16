import pandas as pd


def read_and_write_iris_data(source, destination):
    column_names = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width', 'class']
    data = pd.read_csv(source, delimiter=',', names=column_names)
    data.to_csv(destination, index=False)


if __name__ == "__main__":
    # read iris.data and write to csv format with columns name
    source = 'assets\iris\iris.data'
    destination = 'csv_repository\iris_data.csv'
    read_and_write_iris_data(source, destination)
