import pandas as pd
import numpy as np
from maths import *


def main():
    df = pd.read_csv('dataset_train.csv')
    des_array = np.asarray([])
    for (columnName, columnData) in df.iteritems():
        values = columnData.values
        name_arr = columnName
        count_val = count(values)
        sub_first = np.asarray([name_arr, count_val])
        if type(values[0]) == np.int64 or type(values[0]) == np.float64:
            mean_val = mean(values)
            std_val = standard_deviation(values)
            min_val = min_(values)
            quant_25_val = quantile(values, 0.25)
            quant_50_val = quantile(values, 0.50)
            quant_75_val = quantile(values, 0.75)
            max_val = max_(values)
            sub_array = np.asarray([mean_val, std_val, min_val, quant_25_val, quant_50_val, quant_75_val, max_val])
        else:
            sub_array = np.asarray(['Not numerical data'])
        my_array = np.concatenate((sub_first, sub_array))
        new_array = np.append(des_array, my_array, axis=0)
        des_array = new_array
        print(des_array)

    # print(des_array)
    # print(df.describe())

if __name__ == "__main__":
    main()
