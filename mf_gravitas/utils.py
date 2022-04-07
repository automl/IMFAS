import numpy as np
import os
import warnings

def check_diversity(representation, title, epsilon=0.01):
    """

    :param representation: ndarray.
    :param title: name of the matrix
    :param epsilon: float: the value needed to exceed (should be close to zero)
    :raises: Warning if representation is not diverse
    """
    # Check for (naive) representation collapse by checking sparsity after
    # translation by 90% quantile
    translated = representation - np.quantile(representation, 0.9, axis=0)
    sparsity = (translated < epsilon).sum() / np.product(representation.shape)
    if sparsity >= 0.95:
        warnings.warn(f'The {title} representation is not diverse.')

        # Warning(f'The {title} representation is not diverse.')
        print(representation)

def check_or_create_dir(dir):

    # If folder doesn't exist, then create it.
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print("created folder : ", dir)

    else:
        print(dir, "folder already exists.")