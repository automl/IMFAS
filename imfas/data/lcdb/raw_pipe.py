import lcdb
import numpy as np
import omegaconf
import pandas as pd
from lcdb import *
from matplotlib import pyplot as plt


# fmohr
def plot_curve(anchors, points, ax, color, label=None):
    ax.plot(anchors, [np.median(v) for v in points], color=color, label=label)
    ax.plot(anchors, [np.mean(v) for v in points], linestyle="--", color=color)
    ax.fill_between(anchors, [np.percentile(v, 0) for v in points],
                    [np.percentile(v, 100) for v in points], alpha=0.1, color=color)
    ax.fill_between(anchors, [np.percentile(v, 25) for v in points],
                    [np.percentile(v, 75) for v in points], alpha=0.2, color=color)


# fmohr
def plot_train_and_test_curve(curve, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    anchors = curve[0]
    plot_curve(anchors, curve[1], ax, "C0", label="Performance on Training Data")  # train curve
    plot_curve(anchors, curve[2], ax, "C1",
               label="Performance on Validation Data")  # validation curve
    plot_curve(anchors, curve[3], ax, "C2", label="Performance on Test Data")  # test curve

    ax.plot(anchors,
            [(np.mean(v_train) + np.mean(curve[2][a])) / 2 for a, v_train in enumerate(curve[1])],
            linestyle="--", color="black", linewidth=1)

    ax.axhline(np.mean(curve[2][-1]), linestyle="dotted", color="black", linewidth=1)
    ax.fill_between(anchors, np.mean(curve[2][-1]) - 0.0025, np.mean(curve[2][-1]) + 0.0025,
                    color="black", alpha=0.1, hatch=r"//")

    ax.legend()
    ax.set_xlabel("Number of training instances")
    ax.set_ylabel("Prediction Performance")

    if fig is not None:
        return fig


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
    """
    lcdb
    cfg = omegaconf.DictConfig(kwargs)

    dataset_meta_features = lcdb.get_meta_features()
    dataset_meta_features.index = dataset_meta_features['Name']

    curves = lcdb.get_all_curves(cfg.metric)
    algo_names = set(curves['learner'])
    dataset_ids = set(dataset_meta_features.openmlid)

    lcdb.get_curve()

    fidelity
    anchors, scores_train, scores_valid, scores_test = lcdb.get_curve(
        3,  # openmlid
        "sklearn.linear_model.LogisticRegression",  # learner
        cfg.metric  # metric
    )

    pd.DataFrame(scores_train).T.plot()

    curve = anchors, scores_train, scores_valid, scores_test

    fig = lcdb.plot_train_and_test_curve(curve)
