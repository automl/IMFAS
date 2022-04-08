from itertools import chain

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensemble(df, k, plot_regret_distribution=False):
    """
    Forward selection procedure.
    build a set of candidate algorithms, by choosing the top-k performing algorithms
    for each respective dataset and take the union across datasets.

    :param k: int top-k algorithms that are considered on each dataset respectively
    :return: tuple[set, pd.Dataframe] set of candidates, the respective performance profiles.
    """
    best = df.idxmax()
    candidates = set(best)

    # for each dataset select the k best algorithms separately.
    candidates = set(chain(*[list(df[col].nlargest(k, keep='first').index) for col in df]))

    # look at those candidate's performances across all datasets.
    candidate_performances = df.iloc[list(candidates)]

    if plot_regret_distribution:
        # mean regret on taking this algo against the dataset-respective best algorithm
        regret = candidate_performances.max() - candidate_performances
        # regret.mean(axis=1)

        sns.boxplot(x="variable", y="value", data=pd.melt(regret.transpose()))
        plt.xlabel('Algorithm')
        plt.ylabel('regret against best on dataset')
        plt.xticks(rotation=90)
        plt.show()

    return candidates, candidate_performances


if __name__ == '__main__':
    pass
