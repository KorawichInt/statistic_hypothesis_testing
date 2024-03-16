import pandas as pd

data = pd.read_csv('assets\iris\iris.data', delimiter=' ')
data.to_csv('csv_repository\iris_data.csv', index=False)
# df = pd.DataFrame(data)
