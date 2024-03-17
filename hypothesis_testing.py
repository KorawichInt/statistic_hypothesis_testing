import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import f_oneway


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


def sepal_petal_mean(destination_path):
    df = pd.read_csv(destination_path, header=None)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)
    mean_values = df.iloc[:, :-1].mean()
    mean_list = mean_values.tolist()
    return mean_list


def chi_square_test(observed_df):
    chi2_stat, p_val, dof, expected_counts = chi2_contingency(observed_df)
    return chi2_stat, p_val, dof, expected_counts


def decision_and_conclusion(chi2_stat, p_val, alpha, dof):
    chi_critical = chi2.ppf(1 - alpha, dof)
    if (chi2_stat <= chi_critical) and (p_val >= alpha):
        return (f"Since,\tchi-square value \t\t= {chi2_stat:.4f}\
            \n\tchi-square critical value \t= {chi_critical:.4f}\
            \nand\tp-value \t\t\t= {p_val:.4f}\n\tsignificance coefficient \t= {alpha}\
            \nso,\tchi-square value < chi-square critical value\
            \nalso,\tp-value > significance coefficient\
            \n\nConclusion\n-Fail to reject null hypothesis (H0)\
            \n-This mean there is insufficient evidence to conclude that the proportion of all iris's class are differ.")
    else:
        return (f"Since,\tchi-square value \t\t= {chi2_stat:.4f}\
            \n\tchi-square critical value \t= {chi_critical:.4f}\
            \nand\tp-value \t\t\t= {p_val:.4f}\n\tsignificance coefficient \t= {alpha}\
            \nso,\tchi-square value > chi-square critical value\
            \nalso,\tp-value < significance coefficient\
            \n\nConclusion\n-Reject null hypothesis (H0).\
            \n-This mean there is sufficient evidence to conclude that the proportion of all iris's class are differ.")


def proportion_test_integrated():
    # count number of each classes from 3 group
    training_set_class_count = class_count(destination_path2[0], classes)
    validation_set_class_count = class_count(destination_path2[1], classes)
    testing_set_class_count = class_count(destination_path2[2], classes)
    all_set_dict = {columns[0]: training_set_class_count,
                    columns[1]:  validation_set_class_count,
                    columns[2]: testing_set_class_count}

    observed_df = pd.DataFrame.from_dict(all_set_dict)

    # chi-square test
    chi2_stat, p_val, dof, expected_counts = chi_square_test(observed_df)

    # visualize observed and expected freequency with dataframe
    expected_df = pd.DataFrame(expected_counts, index=classes, columns=columns)
    # print('# Observed Frequency\n', observed_df, '\n')
    # print('# Expected Frequency\n', expected_df, '\n')

    # decision and conclusion
    alpha = 0.05
    asm1_result = decision_and_conclusion(chi2_stat, p_val, alpha, dof)
    return asm1_result


if __name__ == "__main__":
    # define path, parameters, other assets
    print()
    source_path = r'assets\iris\iris.data'
    destination_path = r'csv_repository\iris_data.csv'
    destination_path2 = [r'csv_repository\training_set.csv',
                         r'csv_repository\validation_set.csv',
                         r'csv_repository\testing_set.csv']
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    columns = ['Training_set', 'Validation_set', 'Testing_set']
    feature_sd = {'SepalLength': 0.83, 'SepalWidth': 0.43,
                  'PetalLength': 1.76, 'PetalWidth': 0.76}

    # read iris.data and write to csv format with columns name
    # check if iris_data.csv is not exists
    check_iris_csv = check_file_exists(destination_path)
    if not (check_iris_csv):
        read_and_write_data(source_path, destination_path)

    # check if iris_data.csv is exists and training_set is not exists
    check_training_set = check_file_exists(destination_path2[0])
    if not (check_training_set):
        sampling_data(destination_path, destination_path2)

    """Hypothesis 1: Are proportion of all 3 iris's class equal with significance level 5%"""
    asm1 = proportion_test_integrated()
    print(asm1)

    """Hypothesis 2: Are mean of Sepal length, Sepal width, Petal length and Petal width equal with significance level 5%"""
    print()
    feature_sd_df = pd.DataFrame(list(feature_sd.values()), columns=[
                                 'SD'], index=feature_sd.keys())
    print(feature_sd_df)
    sepal_and_petal = [key for key in feature_sd.keys()]
    print(sepal_and_petal)
    training_set_mean = sepal_petal_mean(destination_path2[0])
    validation_set_mean = sepal_petal_mean(destination_path2[1])
    testing_set_mean = sepal_petal_mean(destination_path2[2])
    all_mean_transpose = list(map(list, zip(*[training_set_mean, validation_set_mean,
                                              testing_set_mean])))
    all_mean_df = pd.DataFrame(all_mean_transpose, columns=columns,
                               index=sepal_and_petal)
    # print(all_mean_df)
