import random


def train_test_split(n, share):
    train_split = random.sample(
        list(range(n)),
        k=int(n * share)
    )

    test_split = list(set(range(n)) - set(train_split))

    return train_split, test_split
