import pandas as pd

column_names = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'class']
data = pd.read_csv('assets\iris\iris.data', delimiter=',', names=column_names)
data.to_csv('csv_repository\iris_data.csv', index=False)
# df = pd.DataFrame(data)
