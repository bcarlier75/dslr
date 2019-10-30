from describe import get_dataset, filter_dataframe
from sys import argv
import matplotlib.pyplot as plt


def plot_scatter(df, column_1, column_2):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[column_1], df[column_2],
                label='Students', color='tab:orange',
                s=20, alpha=0.8)
    plt.legend(loc='upper right')
    plt.title(f'Scatter plot of {column_1} vs {column_2}')
    plt.xlabel(column_1)
    plt.ylabel(column_2)
    plt.show()


def main():
    if len(argv) > 1:
        df_raw = get_dataset(argv[1])
    else:
        return print('Please input the path to the dataset as the first argument.')
    if df_raw is None:
        return print('Please input a valid path to the dataset.')
    df_filter = filter_dataframe(df_raw)
    possible_columnname = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    if len(argv) == 4:
        if argv[2] in possible_columnname and argv[3] in possible_columnname:
            plot_scatter(df_filter, argv[2], argv[3])
            return
    plot_scatter(df_filter, 'Astronomy', 'Defense Against the Dark Arts')
    return


if __name__ == "__main__":
    main()
