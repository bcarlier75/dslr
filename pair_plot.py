from describe import get_dataset
from sys import argv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)


def plot_pairplot(df):
    sns.pairplot(df, hue="Hogwarts House", markers=".", height=2, plot_kws=dict(linewidth=0))
    plt.show()


def main():
    if len(argv) > 1:
        df_raw = get_dataset(argv[1])
    else:
        return print('Please input the path to the dataset as the first argument.')
    if df_raw is None:
        return print('Please input a valid path to the dataset.')
    possible_columnname = ['Hogwarts House', 'Astronomy', 'Herbology',
                           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                           'Transfiguration', 'Potions', 'Charms', 'Flying']
    df = df_raw[possible_columnname]
    df = df.dropna()
    plot_pairplot(df)
    return


if __name__ == "__main__":
    main()
