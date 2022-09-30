import logging
from pathlib import Path

import omegaconf
import torch
from lcdb import get_curve, get_meta_features
from tqdm import tqdm

from imfas.data.lcdb.lcdb_api import LCDB

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# fmohr
import numpy as np


def plot_curve(anchors, points, ax, color, label=None):
    ax.plot(anchors, [np.median(v) for v in points], color=color, label=label)
    ax.plot(anchors, [np.mean(v) for v in points], linestyle="--", color=color)
    ax.fill_between(anchors, [np.percentile(v, 0) for v in points],
                    [np.percentile(v, 100) for v in points], alpha=0.1, color=color)
    ax.fill_between(anchors, [np.percentile(v, 25) for v in points],
                    [np.percentile(v, 75) for v in points], alpha=0.2, color=color)


# # fmohr
# def plot_train_and_test_curve(curve, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = None
#     anchors = curve[0]
#     plot_curve(anchors, curve[1], ax, "C0", label="Performance on Training Data")  # train curve
#     plot_curve(anchors, curve[2], ax, "C1",
#                label="Performance on Validation Data")  # validation curve
#     plot_curve(anchors, curve[3], ax, "C2", label="Performance on Test Data")  # test curve
#
#     ax.plot(anchors,
#             [(np.mean(v_train) + np.mean(curve[2][a])) / 2 for a, v_train in enumerate(curve[1])],
#             linestyle="--", color="black", linewidth=1)
#
#     ax.axhline(np.mean(curve[2][-1]), linestyle="dotted", color="black", linewidth=1)
#     ax.fill_between(anchors, np.mean(curve[2][-1]) - 0.0025, np.mean(curve[2][-1]) + 0.0025,
#                     color="black", alpha=0.1, hatch=r"//")
#
#     ax.legend()
#     ax.set_xlabel("Number of training instances")
#     ax.set_ylabel("Prediction Performance")
#
#     if fig is not None:
#         return fig
#
#
# from lcdb import get_curve
#
# get_curve()
# plot_curve()


def raw_pipe(*args, **kwargs):
    """
    To understand lcdb (at all) one should consult
    https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_1317.pdf
    first -- because the interface of lcdb is not at all documented and just a pile
    of functions (shit).


    relevant quotes from f mohr's paper:
    1. The current version of LCDB provides already over 150GB of ground truth and
    prediction vectors of **20 classification algorithms from the scikit-learn
    library on 246 datasets**. These prediction vectors have been recorded for
    models being trained on an amount of instances (called anchors) that are
    rounded multiples of √ 2. Instead of training each learner only once at
    each anchor, we created 25 splits in order to be able to obtain more
    robust insights into the out-of-sample performance. This makes it, in
    various respects, the biggest database for learning curves available today.

    2. with our learning curves which plot performance versus training set size

    3. LCDB is the first to provide **(probabilistic) predictions** together with the
    ground-truth labels. The availability of probabilistic predictions makes it possible
    to compute one’s own choice of metrics, like AUROC or log-loss, rather then have
    to deal with precomputed ones. Curve data is provided for 25 stratified splits at
    each anchor for training, validation, and test data,

    4. Intended use:
    (i) the a-posteriori analysis of
    the shape of curves for different learners and of function classes describing such
    shapes, (ii) the study of the relationship between training and test curves, (iii)
    the simultaneous analysis of learning curves, e.g., whether or not they intersect
    or if such intersection can be predicted, (iv) research into principled models for
    the runtime behavior of the algorithms, (v) **benchmarking algorithm selection**
    problems, and (vi) quick insights into the “difficulty” of datasets, which can be
    useful for the design of such benchmarks.

    5. training set sizes, which are called anchors

    6. A learning algorithm is a function a : D ×Ω → H , where H = {h | h : X → Y} is the space
    of hypotheses and Ω is a source of randomness, such as the random seed.

    --> this seems to suggest, they have 25 splits at each anchor (and maybe 5 seeds?) -->
    amounting to 5* 25 =125 observation at each anchor (or a total of 125 learning curves?)
    They seem to try to get a distribution over learning curves.

    7. Notice there definition of training set size :
    At anchor s, this estimate is obtained
    by (i) creating one (hold-out) or several (cross-validation) splits (Dtr , Dte ) such
    that |Dtr | = s,



    # notice: when loading this dataset, Algo 13, dataset 91 has mixed precision (and is not
    saved to tensor



    Exploring the the LCDB dataset,
    the following distribution (look at data.eda.lcdb) over available _fidelities across datasets emerged, for 'sklearn.linear_model.LogisticRegression'
    (fidelity, n_datasets), where n_datasets is the #datasets, on which this fidelity is available.
    it seems though (and this is not solid) that at least on a fixed dataset, the fidelity
    is the same across all algorithms (and the same across all splits & seeds)



    """

    cfg = omegaconf.DictConfig(kwargs)

    path_preprocessed = Path(cfg.path_preprocessed)

    # fixme: make the available _fidelities a config option
    lcdb = LCDB(
        path=path_preprocessed,
        fidelities=set.union(*LCDB.available_fidelities[:-3]),
        metric="accuracy"
    )
    # lcdb.collect_all_data()

    dataset_meta_features = get_meta_features()
    dataset_ids = dataset_meta_features['openmlid']

    fidelities = {}
    skipped_odd_shape = {}
    skipped = {}
    error = {}
    for algo in tqdm(lcdb.algo_names, desc='Algos'):
        for dataset in tqdm(dataset_ids, desc='Datasets'):

            try:
                subset_sizes, scores_train, scores_valid, scores_test = get_curve(
                    dataset,  # openmlid
                    algo,  # learner
                )
                fidelities[algo, dataset] = subset_sizes

                try:
                    torch.tensor(scores_train)
                except Exception as e:
                    skipped_odd_shape[algo, dataset] = [len(v) for v in scores_train]
                    print(e)


            except Exception as e:
                print(e)
                skipped[(algo, dataset)] = e

    print(f"skipped {skipped}")
    print(f"skipped_odd_shape {skipped_odd_shape}")
    print(f"fidelities {fidelities}")
# ------------------------------------------------------------------------------
# write the log of the download to file
# fixme: override the log file!
# fh = logging.FileHandler(path_preprocessed / 'download.log')
# fh.setLevel(logging.DEBUG)
# log.addHandler(fh)

# path_train = path_preprocessed / "lcs_train_datasets"
# path_valid = path_preprocessed / "lcs_valid_datasets"
# path_test = path_preprocessed / "lcs_test_datasets"
#
# for path in [path_train, path_valid, path_test]:
#     path.mkdir(parents=True, exist_ok=True)
#
# log.info("Finding dataset_meta_features")
# dataset_meta_features = get_meta_features()
# dataset_meta_features.index = dataset_meta_features['Name']
# dataset_meta_features.to_csv(path_preprocessed / 'dataset_meta_features.csv')
#
# curves = get_all_curves(cfg.metric)
# algo_names = set(curves['learner'])
# del curves
# dataset_ids = set(dataset_meta_features.openmlid)
#
# # Generate the size
# n_spits = 25
# n_seeds = 5
# n_train_size_subsets = 16
#
# # n_datasets, n_algos, n_repetitions, n_fidelities
# shape = (len(algo_names), n_spits * n_seeds, n_train_size_subsets)
#
# anchors = [16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2588]
#
# # some "lcs" are in fact zero or ones tensors
# # ones = torch.ones((125, 16))
# # zeros = torch.zeros((125, 16))
# treat_with_care_ones = []
# treat_with_care_zeros = []
# errors = []
# log.info('Starting to load the Learning Curves')
# for d, dataset_id in tqdm(enumerate(dataset_ids), desc='Datasets', total=len(dataset_ids)):
#
#     train = torch.zeros(shape)
#     valid = torch.zeros(shape)
#     test = torch.zeros(shape)
#
#     for a, algo_name in tqdm(enumerate(algo_names), desc='Algorithms', total=len(algo_names)):
#         # TODO FUTURE: for algo & dataset major, simply move this part into a separate function,
#         #  and decide and sweep over the files from the other direction.
#
#         try:
#             subset_sizes, scores_train, scores_valid, scores_test = get_curve(
#                 dataset_id,  # openmlid
#                 algo_name,  # learner
#                 cfg.metric  # metric
#             )
#
#             assert len(subset_sizes) == len(
#                 scores_train), f"len(subset_sizes) != len(scores_train) for {dataset_id}, {algo_name}"
#             assert len(subset_sizes) == len(
#                 scores_valid), f"len(subset_sizes) != len(scores_valid) for {dataset_id}, {algo_name}"
#             assert len(subset_sizes) == len(
#                 scores_test), f"len(subset_sizes) != len(scores_test) for {dataset_id}, {algo_name}"
#
#             # this does not hold anyways:
#             # assert subset_sizes == anchors, f"subset_sizes != anchors: {subset_sizes} != {anchors}"
#
#         except Exception as e:
#             log.debug(f"skipping {dataset_id} {algo_name} because of \n{e}")
#             raised.append((dataset_id, algo_name))
#
#             errors[(dataset_id, algo_name)] = e
#             continue
#
#         tr = torch.tensor(scores_train).T
#         va = torch.tensor(scores_valid).T
#         te = torch.tensor(scores_test).T
#
#         # for lcs in [tr, va, te]:
#         #     if torch.equal(lcs, ones):
#         #         treat_with_care_ones.append((dataset_id, algo_name))
#         #     if torch.equal(lcs, zeros):
#         #         treat_with_care_zeros.append((dataset_id, algo_name))
#
#         train[a, :, :] = tr
#         valid[a, :, :] = va
#         test[a, :, :] = te
#
#     # log.info(f"TrainTensor: {train.shape}, of memory footprint: {sys.getsizeof(train)}")
#     torch.save(train, path_train / f"{str(d)}.pt")
#     torch.save(valid, path_valid / f"{str(d)}.pt")
#     torch.save(test, path_test / f"{str(d)}.pt")
#
# log.warning(f'\n\nones= {treat_with_care_ones}\n\n'
#             f'zeros= {treat_with_care_zeros}\n\n'
#             f'Raised= {errors.keys()}\n\n'
#             f'errors= {errors}\n\n')
#
# log.info(f"Done. Saved lcdb to {path_preprocessed}")

# logging.shutdown()

# curve = anchors, scores_train, scores_valid, scores_test
#
# fig = lcdb.plot_train_and_test_curve(curve)

#
# if __name__ == '__main__':
#     ones: [(3, 'sklearn.tree.DecisionTreeClassifier'), (3, 'sklearn.tree.ExtraTreeClassifier'),
#            (3, 'sklearn.ensemble.ExtraTreesClassifier')]
#     zeros = []
#
#     from imfas.data.lcdb.raised import raised
#
#     # analysis on the count of raised
#     D = {k: 0 for k, v in raised}
#     A = {v: 0 for k, v in raised}
#     for k, v in raised:
#         D[k] += 1
#         A[v] += 1
#
#     # prepare for re-run
#     r = {k: [] for k, v in raised}
#     for k, v in raised:
#         r[k].append(v)
#
#     curve = get_curve(*(40996, 'sklearn.naive_bayes.MultinomialNB'))
#
#     # distribution over subset sizes (anchors) for a fixed algorithm
#     D = {}
#     l = []
#     errors = {}
#     for d in tqdm(dataset_ids):
#         try:
#             subset_sizes, scores_train, scores_valid, scores_test = get_curve(
#                 d,
#                 'sklearn.linear_model.LogisticRegression'
#             )
#         except Exception as e:
#             print(f'failed for {d}, \n{e}')
#             errors[(d,)] = e
#             l.append(d)
#
#         D[d] = subset_sizes
#
#     D_last = {k: v[-1] for k, v in D.items()}
#
#     # distribution over subset sizes (anchors) for a fixed dataset
#     A = {}
#     l1 = []
#     for a in tqdm(algo_names):
#         try:
#             subset_sizes, scores_train, scores_valid, scores_test = get_curve(
#                 40996,
#                 a
#             )
#         except Exception as e:
#             print(f'failed for {a}, \n{e}')
#             l1.append(a)
#
#         A[a] = subset_sizes
#
#     A_last = {k: v[-1] for k, v in A.items()}
#
#
#
# scores_train = [[1.0, 0.875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#                 [0.9565, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#                  1.0, 1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9688,
#                  0.9688, 1.0, 1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9333, 1.0, 1.0, 0.9778, 1.0, 1.0, 1.0,
#                  1.0, 1.0, 1.0, 0.9333, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#                 [0.9531, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9844, 1.0, 1.0, 1.0, 1.0,
#                  1.0, 1.0, 1.0, 0.9844, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9844],
#                 [1.0, 0.989, 1.0, 1.0, 1.0, 1.0, 0.989, 1.0, 1.0, 1.0, 1.0, 0.967, 0.967, 0.989,
#                  1.0, 1.0, 0.989, 1.0, 0.967, 1.0, 1.0, 1.0, 1.0, 1.0, 0.989],
#                 [0.9844, 0.9844, 1.0, 1.0, 0.9844, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9922,
#                  0.9844, 1.0, 1.0, 0.9922, 0.9922, 0.9922, 1.0, 0.9922, 0.9922, 1.0, 1.0, 0.9922],
#                 [1.0, 0.9945, 1.0, 0.9945, 0.9613, 0.9945, 1.0, 0.989, 1.0, 1.0, 1.0, 0.989, 0.9669,
#                  0.989, 1.0, 0.989, 1.0, 1.0, 0.989, 0.9834, 1.0, 1.0, 0.9945, 1.0, 0.989],
#                 [0.9609, 1.0, 0.9961, 0.9922, 0.9727, 1.0, 0.9844, 1.0, 0.9453, 1.0, 0.9844, 0.9609,
#                  0.9922, 0.9766, 0.9648, 0.9414, 0.9961, 0.9648, 0.9766, 1.0, 0.9922, 0.9961,
#                  0.9961, 1.0, 0.9922],
#                 [0.9613, 1.0, 0.9834, 0.9917, 0.9945, 0.9945, 0.9972, 1.0, 0.9779, 0.9972, 0.9945,
#                  0.989, 0.989, 0.9834, 0.9834, 0.9779, 0.9834, 1.0, 0.989, 1.0, 0.9862, 0.9779,
#                  0.9945, 1.0, 0.989],
#                 [0.9902, 0.9844, 0.998, 0.9863, 0.9922, 0.9824, 0.9785, 1.0, 0.9902, 0.9941, 0.9902,
#                  0.9922, 0.9844, 0.9883, 0.9961, 0.9727, 0.9961, 0.9844, 0.9902, 0.9961, 0.9922,
#                  0.9805, 0.9824, 1.0, 0.9902],
#                 [0.9848, 0.9765, 0.9917, 0.989, 0.989, 0.9848, 0.9862, 0.9917, 0.9779, 0.9945,
#                  0.9848, 0.9903, 0.9779, 0.989, 0.989, 0.9903, 0.9669, 0.989, 0.9862, 0.989, 0.989,
#                  0.9669, 0.9807, 0.9945, 0.9876],
#                 [0.9912, 0.9873, 0.9922, 0.9854, 0.9863, 0.9805, 0.9697, 0.9893, 0.9805, 0.9932,
#                  0.9951, 0.9863, 0.9795, 0.9736, 0.9912, 0.9785, 0.9863, 0.9941, 0.9824, 0.9863,
#                  0.9824, 0.9824, 0.9609, 0.9805, 0.9902],
#                 [0.9834, 0.9834, 0.98, 0.9876, 0.9717, 0.9848, 0.9682, 0.9834, 0.9786, 0.9869,
#                  0.9841, 0.9807, 0.9876, 0.9903, 0.9869, 0.9551, 0.989, 0.9862, 0.9869, 0.9834,
#                  0.9896, 0.9876, 0.9876, 0.9917, 0.9896],
#                 [0.9697, 0.9858, 0.9868, 0.9888, 0.9854, 0.9668, 0.9795, 0.9863, 0.9697, 0.9463,
#                  0.9067, 0.9604, 0.9893, 0.9849, 0.9868, 0.9814, 0.9897, 0.9863, 0.9858, 0.9883,
#                  0.9839, 0.9746, 0.9834, 0.9927, 0.9893],
#                 [0.9838, 0.9858, 0.9869, 0.9827, 0.9838, 0.9845, 0.9606, 0.9876, 0.9686, 0.9858,
#                  0.9769, 0.9769, 0.9903, 0.9869, 0.9852, 0.9365, 0.9841, 0.9848, 0.9734, 0.9679,
#                  0.9848, 0.9838, 0.9827, 0.9838, 0.9865],
#                 [0.9568, 0.9824, 0.9602, 0.9878, 0.9819, 0.9849, 0.9805, 0.9846, 0.9812, 0.9807,
#                  0.9863, 0.9827, 0.9871, 0.981, 0.9395, 0.9849, 0.9844, 0.9678, 0.9878, 0.98,
#                  0.9858, 0.9839, 0.9639, 0.9832, 0.9819],
#                 [0.9869, 0.9843, 0.9853, 0.9757, 0.9758, 0.9867, 0.9634, 0.9807, 0.986, 0.9698,
#                  0.9884, 0.9776, 0.9853, 0.9853, 0.9869, 0.9824, 0.9867, 0.9857, 0.9869, 0.9729,
#                  0.9717, 0.982, 0.9796, 0.9853, 0.9853],
#                 [0.9696, 0.9854, 0.9617, 0.9739, 0.9854, 0.986, 0.9852, 0.9819, 0.9857, 0.9862,
#                  0.9819, 0.984, 0.9845, 0.9589, 0.9865, 0.9857, 0.9862, 0.9811, 0.98, 0.9702,
#                  0.9852, 0.9709, 0.9843, 0.9865, 0.9851],
#                 [0.9814, 0.9843, 0.9835, 0.9846, 0.9834, 0.9846, 0.969, 0.9849, 0.9749, 0.9824,
#                  0.9761, 0.9836, 0.9721, 0.9784, 0.9839, 0.9839, 0.9812, 0.9857, 0.9845, 0.9836,
#                  0.9829, 0.9815, 0.9864, 0.9853, 0.9864]]
#
