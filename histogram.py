from maths import min_, max_
from describe import get_dataset, filter_dataframe
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['Index'], axis=1)
    for columnName in df.columns:
        df[columnName] = (df[columnName] - df[columnName].mean()) / df[columnName].std()
    return df


def get_grades(df, df_norm, house_name, course_name) -> np.array:
    output_df = df_norm[df["Hogwarts House"] == house_name][course_name]
    return [x for x in output_df[~np.isnan(output_df)]]


def plot_one(df, df_norm, column_name):
    bins = np.linspace(min_(df_norm[column_name]), max_(df_norm[column_name]), 100)
    plt.figure(figsize=(10, 6))
    plt.hist(get_grades(df, df_norm, "Ravenclaw", column_name),
             bins=bins, alpha=0.5, label='Ravenclaw', color='tab:orange')
    plt.hist(get_grades(df, df_norm, "Slytherin", column_name),
             bins=bins, alpha=0.5, label='Slytherin', color='tab:green')
    plt.hist(get_grades(df, df_norm, "Hufflepuff", column_name),
             bins=bins, alpha=0.5, label='Hufflepuff', color='turquoise')
    plt.hist(get_grades(df, df_norm, "Gryffindor", column_name),
             bins=bins, alpha=0.5, label='Gryffindor', color='tab:red')
    plt.legend(loc='upper right')
    plt.title(f'Histogram of "{column_name}" grades among the different Hogwarts houses')
    plt.show()


def plot_histogram(df, df_norm, specified_name=''):
    if specified_name:
        plot_one(df, df_norm, specified_name)
    else:
        for column_name in df_norm.columns:
            plot_one(df, df_norm, column_name)


def main():
    df_raw = get_dataset('datasets/dataset_train.csv')
    if df_raw is None:
        return print('Please input a valid file as first argument.')
    df_filter = filter_dataframe(df_raw)
    df_norm = normalize_dataframe(df_filter)
    possible_columnname = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    if len(argv) > 1:
        if argv[1] in possible_columnname:
            plot_histogram(df_raw, df_norm, argv[1])
            return
    plot_histogram(df_raw, df_norm)
    return


if __name__ == "__main__":
    main()
