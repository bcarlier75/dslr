import pandas as pd
import os
import sys
from maths import *
import numpy as np


def get_dataset():
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        if os.path.exists(file_name) and os.path.isfile(file_name):
            dataframe = pd.read_csv(file_name)
            return dataframe
    return


def filter_dataframe(df_raw):
    df_raw = df_raw.drop(['Hogwarts House'], axis=1)
    df = df_raw.select_dtypes([np.number])
    return df


def manual_describe(df):
    output_df = pd.DataFrame(columns=[columnName for (columnName, columnData) in df.iteritems()],
                             index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    for (columnName, columnData) in df.iteritems():
        if columnName in output_df.columns:
            my_values = [x for x in columnData.values[~np.isnan(columnData.values)]]
            my_values.sort()
            count_val = count(my_values)
            mean_val = mean(my_values)
            std_val = standard_deviation(my_values)
            min_val = min_(my_values)
            quant_25_val = quantile(my_values, 0.25)
            quant_50_val = median(my_values)
            quant_75_val = quantile(my_values, 0.75)
            max_val = max_(my_values)
            output_df[columnName] = [count_val, mean_val, std_val, min_val, quant_25_val,
                                     quant_50_val, quant_75_val, max_val]
    return output_df


def main():
    df_raw = get_dataset()
    if df_raw is None:
        return print('Please input a valid file as first argument.')
    df = filter_dataframe(df_raw)
    describe_output = manual_describe(df)
    print(describe_output)
    print(df.describe())


if __name__ == "__main__":
    main()
