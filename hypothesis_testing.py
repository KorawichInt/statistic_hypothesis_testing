import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2


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


def class_count(destination_path, classes):
    column_names = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Class']
    dataset = pd.read_csv(destination_path, header=None)
    dataset.columns = column_names
    class_counts = dataset['Class'].value_counts()
    count_dict = class_counts.to_dict()
    count_dict = {class_name: count_dict[class_name] for class_name in classes}
    return count_dict


def chi_square_test(observed_df):
    chi2_stat, p_val, dof, expected_counts = chi2_contingency(observed_df)
    return chi2_stat, p_val, dof, expected_counts


# def decision_and_conclusion():
#     print("Chi-square statistic:", chi2_stat)
#     chi_critical = chi2.ppf(1 - alpha, dof)
#     print("Chi-square critical value:", chi_critical)
#     # if
#     print("\nP-value:", p_val)
#     print("Siginifance Coefficient =", alpha)


if __name__ == "__main__":
    # define path
    source_path = r'assets\iris\iris.data'
    destination_path = r'csv_repository\iris_data.csv'
    destination_path2 = [r'csv_repository\training_set.csv',
                         r'csv_repository\validation_set.csv',
                         r'csv_repository\testing_set.csv']

    # read iris.data and write to csv format with columns name
    # check if iris_data.csv is not exists
    check_iris_csv = check_file_exists(destination_path)
    if not (check_iris_csv):
        read_and_write_data(source_path, destination_path)

    # check if iris_data.csv is exists and training_set is not exists
    check_training_set = check_file_exists(destination_path2[0])
    if not (check_training_set):
        sampling_data(destination_path, destination_path2)

    # count number of each classes from 3 group
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    training_set_class_count = class_count(destination_path2[0], classes)
    validation_set_class_count = class_count(destination_path2[1], classes)
    testing_set_class_count = class_count(destination_path2[2], classes)
    all_set_dict = {'Traning_set': training_set_class_count,
                    'Validation_set':  validation_set_class_count,
                    'Testing_set': testing_set_class_count}

    # visualize observed freequency with dataframe
    observed_df = pd.DataFrame.from_dict(all_set_dict)
    print('\n# Observed Frequency\n', observed_df, '\n')

    # chi-square test
    chi2_stat, p_val, dof, expected_counts = chi_square_test(
        observed_df)

    # visualize expected freequency with dataframe
    p_columns = ['Training_set', 'Validation_set', 'Testing_set']
    expected_df = pd.DataFrame(
        expected_counts, index=classes, columns=p_columns)
    print('# Expected Frequency\n', expected_df, '\n')

    # decision and conclusion
    # alpha = 0.05
