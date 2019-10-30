from maths import min_, max_
from describe import get_dataset, filter_dataframe
from sys import argv
import matplotlib.pyplot as plt
import numpy as np


def normalize_dataframe(df):
    df = df.drop(['Index'], axis=1)
    for columnName in df.columns:
        df[columnName] = (df[columnName] - df[columnName].mean()) / df[columnName].std()
    return df


def get_grades(df, norm_df, house_name, course_name):
    output_df = norm_df[df["Hogwarts House"] == house_name][course_name]
    return [x for x in output_df[~np.isnan(output_df)]]


def plot_one(df, norm_df, column_name):
    bins = np.linspace(min_(norm_df[column_name]), max_(norm_df[column_name]), 100)
    plt.figure(figsize=(13, 8))
    plt.hist(get_grades(df, norm_df, "Gryffindor", column_name),
             bins=bins, alpha=0.5, label='Gryffindor', color='tab:red')
    plt.hist(get_grades(df, norm_df, "Ravenclaw", column_name),
             bins=bins, alpha=0.5, label='Ravenclaw', color='tab:orange')
    plt.hist(get_grades(df, norm_df, "Slytherin", column_name),
             bins=bins, alpha=0.5, label='Slytherin', color='tab:green')
    plt.hist(get_grades(df, norm_df, "Hufflepuff", column_name),
             bins=bins, alpha=0.5, label='Hufflepuff', color='turquoise')
    plt.legend(loc='upper right')
    plt.title(f'Histogram of "{column_name}" grades among the different Hogwarts houses')
    plt.show()


def plot_histogram(df, norm_df, specified_name=''):
    if specified_name:
        plot_one(df, norm_df, specified_name)
    else:
        for column_name in norm_df.columns:
            plot_one(df, norm_df, column_name)


def main():
    df_raw = get_dataset('datasets/dataset_train.csv')
    if df_raw is None:
        return print('Please input a valid file as first argument.')
    filter_df = filter_dataframe(df_raw)
    norm_df = normalize_dataframe(filter_df)
    possible_columnname = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    if len(argv) > 1:
        if argv[1] in possible_columnname:
            plot_histogram(df_raw, norm_df, argv[1])
            return
    plot_histogram(df_raw, norm_df)
    return


if __name__ == "__main__":
    main()
