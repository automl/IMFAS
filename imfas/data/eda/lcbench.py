import math
import matplotlib.pyplot as plt
import pandas as pd

from imfas.data.dataset_join.dataset_join_Dmajor import Dataset_Join_Dmajor


def plot_dataset_curves(dataset: Dataset_Join_Dmajor, idx: int, ax=None, major='D'):
    if major == 'D':
        lcs = dataset.lcs.transformed_df[idx, :, :]
    elif major == 'A':
        lcs = dataset.lcs.transformed_df[:, idx, :]

    pd.DataFrame(lcs.numpy().T).plot(legend=False, ax=ax)


def plot_alldataset_curves(dataset: Dataset_Join_Dmajor, dataset_name, major='D', **kwargs):
    if major == 'D':
        n = len(dataset)
    elif major == 'A':
        n = dataset.lcs.shape[1]  # number of algorithms

    r = math.ceil(math.sqrt(n))
    fig, axs = plt.subplots(ncols=r,
                            nrows=r,
                            figsize=(r * 1.5, r * 1.5),
                            layout="constrained",
                            sharex=True, **kwargs)

    for i, ax in zip(range(n), axs.flatten()):
        plot_dataset_curves(dataset, i, ax, major=major)

    for ax in axs.flatten()[i:]:
        ax.set_visible(False)

    fig.suptitle(f"Dataset {dataset_name} - {major} major")


# TODO : make this a method of DatasetLC?


if __name__ == '__main__':
    from imfas.data.lcbench.example_data import train_dataset

    # look at the whole dataset
    train_dataset.split = list(range(35))

    plot_dataset_curves(train_dataset, 3, major='D')
    plt.show()
    plot_alldataset_curves(train_dataset, dataset_name='LCBench', sharey=False, major='D')
    plt.show()
