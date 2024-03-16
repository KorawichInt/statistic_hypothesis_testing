import pandas as pd
import os
from scipy import stats


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
    validation_set = data_shuffled[group_sizes[0]
        :group_sizes[0]+group_sizes[1]]
    testing_set = data_shuffled[group_sizes[0]+group_sizes[1]:]
    training_set.to_csv(destination_path2[0], index=False, header=False)
    validation_set.to_csv(destination_path2[1], index=False, header=False)
    testing_set.to_csv(destination_path2[2], index=False, header=False)


def class_proportion(destination_path, classes):
    column_names = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Class']
    dataset = pd.read_csv(destination_path, header=None)
    dataset.columns = column_names
    class_counts = dataset['Class'].value_counts()
    count_list = class_counts.values.tolist()
    proportion_list = [c/sum(count_list) for c in count_list]
    return proportion_list


def proportion_ztest(mean_proportion, alpha):
    stat, p_value = stats.proportions_ztest(mean_proportion)
    print("Z-score:", stat)
    print("P-value:", p_value)
    alpha = alpha  # Significance level
    if p_value < alpha:
        print("Reject the null hypothesis: Proportions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: Proportions are not significantly different.")


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

    # check if iris_data.csv is exists and training_set is not exists
    check_training_set = check_file_exists(destination_path2[0])
    if not (check_training_set):
        print("2")
        sampling_data(destination_path, destination_path2)

    # find proportion of these classes
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    trainning_set_proportion = class_proportion(destination_path2[0], classes)
    print("\n# Training set proportion")
    for i in range(len(classes)):
        print(f"{classes[i]} proportion = {trainning_set_proportion[i]}")
    validation_set_proportion = class_proportion(destination_path2[1], classes)
    print("\n# Validation set proportion")
    for i in range(len(classes)):
        print(f"{classes[i]} proportion = {validation_set_proportion[i]}")
    testing_set_proportion = class_proportion(destination_path2[2], classes)
    print("\n# Training set proportion")
    for i in range(len(classes)):
        print(f"{classes[i]} proportion = {testing_set_proportion[i]}")

    print("\n# Mean Proportion")
    mean_proportion = [(train + validate + test)/3 for train, validate, test in zip(
        trainning_set_proportion, validation_set_proportion, testing_set_proportion)]
    for i in range(len(classes)):
        print(f"{classes[i]} proportion mean = {mean_proportion[i]}")

    # proportion z test
    alpha = 0.05
    proportion_ztest(mean_proportion, alpha)
