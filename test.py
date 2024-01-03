import pandas as pd
import numpy as np
import warnings
from collections import Counter

warnings.simplefilter(action='ignore', category=FutureWarning)

names = [
    'sex',
    'length',
    'diameter',
    'height',
    'whole_weight',
    'shucked_weight',
    'viscera_weight',
    'shell_weight',
    'rings',
]

data = pd.read_csv('abalone/abalone.data', names=names)

print('there is min value of 0 of height that needs to be replaced')
print(data.describe())
print('---------------------------------------------')

print('will show 2 records with height = 0')
print(data[data['height'] == 0])
print('---------------------------------------------')

print('all data with height != 0')
data_height_not_zero = data[data['height'] > 0]
print(data_height_not_zero.describe())
print('---------------------------------------------')

print('means from all records except 2 with zeros')
means = pd.pivot_table(data_height_not_zero, index=['sex'], aggfunc={'height': np.mean})
mean_for_infant = means.at['I', 'height']
print(means)
print('---------------------------------------------')

# data without zeros
data['height'] = data['height'].replace(to_replace=0, value=mean_for_infant)

def find_and_delete_outliers(df):
    print(len(df))
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)

    IQR = Q3 - Q1

    outliers_mask = ((df[numeric_columns] <= (Q1 - 1.5 * IQR)) | (df[numeric_columns] >= (Q3 + 1.5 * IQR))).any(axis=1)

    return df[~outliers_mask]


data_without_sex = data.drop('sex', axis=1)

t = find_and_delete_outliers(data_without_sex)
print(t.describe())


