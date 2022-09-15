import time

import scipy
import numpy as np
import pandas as pd
import torch

from lccv.lccv import EmpiricalLearningModel


class LCCVELM(EmpiricalLearningModel):
    """
    This is a variatio of the original EmpiricalLearningModel's implementation.
    The main change is that, since we have the learning curve values, we could directly replace all the error-rate
    values in the raw EmpiricalLearningModel with the available leearning curve values. Additionally,
    IMFAS does not consider running time. I will delete all the . We simply set running time as the run time.

    I replace all error_rate related stuff with lc_values
    """
    def __init__(self, algo_idx: int, seed):
        self.algo_idx = algo_idx
        self.active_seed = seed
        self.df = pd.DataFrame([], columns=["trainsize", "seed", "lc_value", "runtime"])

    def evaluate(self, lc: torch.Tensor, size, timeout, verbose):
        # FIXME I simply ignore batch setting here, as this approch might be hard to be parallized
        return lc[size, self.algo_idx]

    def compute_and_add_sample(self, lc: torch.Tensor, size, seed=None, timeout=None, verbose=False):
        tic = time.time()
        # TODO: important to check whether this is always a different order
        lc_value = self.evaluate(lc,  size,
            timeout / 1000 if timeout is not None else None, verbose)
        toc = time.time()
        runtime = int(np.round(1000 * (toc - tic)))
        self.logger.debug(f"Sample value computed within {runtime}ms")
        self.df.loc[len(self.df)] = [size, seed, lc_value, size]
        self.df = self.df.astype({"trainsize": int, "seed": int, "runtime": int})
        return lc_value

    def get_values_at_anchor(self, anchor, test_scores=True):
        return self.df[self.df["trainsize"] == anchor]["lc_values"].values

    def get_best_worst_train_score(self):
        return max([min(g) for i, g in self.df.groupby("trainsize")["lc_values"]])

    def get_ipl(self):
        sizes = sorted(list(pd.unique(self.df["trainsize"])))
        scores = [np.mean(self.df[self.df["trainsize"] == s]["lc_value"]) for s in sizes]

        def ipl(beta):
            a, b, c = tuple(beta.astype(float))
            pl = lambda x: a + b * x ** (-c)
            penalty = []
            for i, size in enumerate(sizes):
                penalty.append((pl(size) - scores[i]) ** 2)
            return np.array(penalty)

        a, b, c = tuple(scipy.optimize.least_squares(ipl, np.array([1, 1, 1]), method="lm").x)
        return lambda x: a + b * x ** (-c)

    def get_mmf(self, validation_curve=True):
        sizes = sorted(list(pd.unique(self.df["trainsize"])))
        scores = [np.mean(self.df[self.df["trainsize"] == s]["lc_value"])
                  for s in sizes]
        weights = [2 ** i for i in range(len(sizes))]

        def mmf(beta):
            a, b, c, d = tuple(beta.astype(float))
            fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
            penalties = []
            for i, size in enumerate(sizes):
                penalty = weights[i] * ((scores[i] - fun(size)) ** 2)  # give more weights on higher anchors
                penalties.append(penalty if not np.isnan(penalty) else 10 ** 6)
            return sum(penalties)

        factor = 1 if validation_curve else -1
        const = {
            "type": "ineq", "fun": lambda x: -factor * x[1] * (x[2] - x[0]) * x[3],
            # "type": "ineq", "fun": lambda x: factor if all([(x[2] - x[0]) * ((x[3] + 1)* size**x[3] - x[1]*x[3] + x[2]) for size in np.linspace(64, 10000, 1000)]) else -factor
        }

        a, b, c, d = tuple(scipy.optimize.minimize(mmf, np.array([0.5, 1, 1, -1]), constraints=const).x)
        return (a, b, c, d), lambda x: (a * b + c * x ** d) / (b + x ** d)

    def get_normal_estimates(self, size=None, round_precision=100, validation=True):

        if size is None:
            sizes = sorted(np.unique(self.df["trainsize"]))
            out = {}
            for size in sizes:
                out[int(size)] = self.get_normal_estimates(size)
            return out

        dfProbesAtSize = self.df[self.df["trainsize"] == size]
        mu = np.mean(dfProbesAtSize["lc_values"])
        sigma = np.std(dfProbesAtSize["lc_values"])
        return {
            "n": len(dfProbesAtSize["lc_values"]),
            "mean": np.round(mu, round_precision),
            "std": np.round(sigma, round_precision),
            "conf": np.round(
                scipy.stats.norm.interval(0.95, loc=mu, scale=sigma / np.sqrt(len(dfProbesAtSize))) if sigma > 0 else (
                mu, mu), round_precision)
        }


def lccv(algo_idx, lc, n_slices: int, r=1.0, timeout=None, base=2, min_exp=6, MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION=0.005,
         MAX_EVALUATIONS=10, target_anchor=.9, return_estimate_on_incomplete_runs=False,
         max_conf_interval_size_default=0.1, max_conf_interval_size_target=0.001, enforce_all_anchor_evaluations=False,
         seed=0, verbose=False, logger=None, min_evals_for_stability=1, use_train_curve=False,
         fix_train_test_folds=False, visualize_lcs=False):
    """
    TODO Given that we have lC values. I simply select all the important values that matters here:

    Evaluates a learner in an iterative fashion, using learning curves. The
    method builds upon the assumption that learning curves are convex. After
    each iteration, it checks whether the convexity assumption is still valid.
    If not, it tries to repair it.
    Also, after each iteration it checks whether the performance of the best
    seen learner so far is still reachable by making an optimistic extrapolation.
    If not, it stops the evaluation.

    :param algo_idx: The index of the algorithm that we want to predict with lccv
    :param lc: learning curves, could be directly extraced from our datasets
    :param n_slices: The number of slices (length of learning curves) FIXME: THis could also be the length of the entire learning curves
    :param r: The best seen performance so far (lower is better). Fill in 0.0 if
    no learners have been evaluated prior to the learner.

    #FIXME the following parameters determine a exponential schedule. We could change it to something else
    :param base: The base factor to increase the sample sizes of the learning
    curve.'
    :param min_exp: The first exponent of the learning curve.

    #FIXME THE following parameters determine the stopping criterion. We need to consider how to compare it properly with our METHODS!!!!
    :param MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION: The maximum number of
    evaluations to be performed
    :param MAX_EVALUATIONS:
    :param target_anchor: The timestep we want to stop or predict

    #FIXME the following arguments
    :param return_estimate_on_incomplete_runs:
    :param max_conf_interval_size_default:
    :param max_conf_interval_size_target:
    :param enforce_all_anchor_evaluations:
    :param seed:
    :param verbose:
    :param logger:
    :param min_evals_for_stability:
    :param use_train_curve: If True, then the evaluation stops as soon as the train curve drops under the threshold r
    :return:
    """
    # intialize
    tic = time.time()
    deadline = tic + timeout if timeout is not None else None

    # initialize important variables and datastructures
    max_exp = np.log(target_anchor) / np.log(base)
    # TODO
    schedule = range(list(n_slices))
    # schedule = [base**i for i in list(range(min_exp, int(np.ceil(max_exp))))] + [target_anchor]
    slopes = (len(schedule) - 1) * [np.nan]
    elm = LCCVELM(algo_idx, seed)
    T = len(schedule) - 1
    t = 0 if r < 1 or enforce_all_anchor_evaluations else T
    repair_convexity = False

    # announce start event together with state variable values
    logger.info(f"""Running LCCV on {X.shape}-shaped data. Overview:
    r: {r}
    min_exp: {min_exp}
    max_exp: {max_exp}
    Seed is {seed}
    t_0: {t}
    Schedule: {schedule}""")

    ## MAIN LOOP
    while t <= T and elm.get_conf_interval_size_at_target(target_anchor) > max_conf_interval_size_target and len(
            elm.get_values_at_anchor(target_anchor)) < MAX_EVALUATIONS:

        # We don't use
        remaining_time = deadline - time.time() if deadline is not None else np.inf
        if remaining_time < 1:
            logger.info("Timeout observed, stopping outer loop of LCCV")
            break

        # initialize stage-specific variables
        eps = max_conf_interval_size_target if t == T else max_conf_interval_size_default
        s_t = schedule[t]
        num_evaluations_at_t = len(elm.get_values_at_anchor(s_t))
        logger.info(f"Running iteration for t = {t}. Anchor point s_t is {s_t}. Remaining time: {remaining_time}s")

        ## INNER LOOP: acquire observations at anchor until stability is reached, or just a single one to repair convexity
        while repair_convexity or num_evaluations_at_t < min_evals_for_stability or (
                elm.get_conf_interval_size_at_target(s_t) > eps and num_evaluations_at_t < MAX_EVALUATIONS):

            remaining_time = deadline - time.time() if deadline is not None else np.inf
            if remaining_time < 1:
                logger.info("Timeout observed, stopping inner loop of LCCV")
                break

            # unset flag for convexity repair
            repair_convexity = False

            # compute next sample
            try:
                seed_used = 13 * (1 + seed) + num_evaluations_at_t
                logger.debug(f"Adding point at size {s_t} with seed is {seed_used}. Remaining time: {remaining_time}s")
                # TODO make this a seperate function that allows us to pass lc here!!!
                error_rate_train, error_rate_test = elm.compute_and_add_sample(lc, s_t, seed_used, (
                                                                                                           deadline - time.time()) * 1000 if deadline is not None else None,
                                                                               verbose=verbose)
                num_evaluations_at_t += 1
                logger.debug(
                    f"Sample computed successfully. Observed performance was {np.round(error_rate_train, 4)} (train) and {np.round(error_rate_test, 4)} (test).")
            except Exception as e:
                logger.info(
                    f"Observed an exception at anchor {s_t}.\nRaising it to the outside and ignoring this candidate.\nThis is not necessarily a good strategy; depending on the exception, one should try the candidate again on the same or bigger data size, because this can be related to a too small sample size.\nThe exception was: {e}.")
                raise

            # check wheter a repair is needed
            if num_evaluations_at_t >= min_evals_for_stability and t < T:
                if visualize_lcs and t > 2:
                    elm.visualize(schedule[-1], r)

                    slopes = elm.get_slope_ranges()
                    if len(slopes) < 2:
                        raise Exception(
                            f"There should be two slope ranges for t > 2 (t is {t}), but we observed only 1.")
                    if slopes[t - 2] < slopes[t - 1] and len(
                            elm.get_values_at_anchor(schedule[t - 1])) < MAX_EVALUATIONS:
                        repair_convexity = True
                        break

        # check training curve
        if use_train_curve != False:

            check_training_curve = (type(use_train_curve) == bool)
            # TODO not sure how use_train_curveis used
            # or (callable(use_train_curve) and use_train_curve(learner_inst, s_t))

            if check_training_curve and elm.get_best_worst_train_score() > r:
                logger.info(
                    f"Train curve has value {elm.get_best_worst_train_score()} that is already worse than r = {r}. Stopping.")
                break

        # after the last stage, we dont need any more tests
        if t == T:
            logger.info("Last iteration has been finished. Not testing anything else anymore.")
            break

        # now decide how to proceed
        if repair_convexity:
            t -= 1
            logger.debug(f"Convexity needs to be repaired, stepping back. t is now {t}")
        elif t >= 2 and elm.get_performance_interval_at_target(target_anchor)[1] >= r:
            optimistic_estimate_for_target_performance = elm.get_performance_interval_at_target(target_anchor)[1]

            # prepare data for cut-off summary
            pessimistic_slope, optimistic_slope = elm.get_slope_range_in_last_segment()
            estimates = elm.get_normal_estimates()
            sizes = sorted(np.unique(elm.df["trainsize"]))
            i = -1
            while len(elm.df[elm.df["trainsize"] == sizes[i]]) < 2:
                i -= 1
            last_size = s_t
            normal_estimates_last = estimates[last_size]
            last_conf = normal_estimates_last["conf"]

            # inform about cut-off
            logger.info(
                f"Impossibly reachable. Best possible score by bound is {optimistic_estimate_for_target_performance}. Stopping after anchor s_t = {s_t} and returning nan.")
            logger.debug(f"""Details about stop:
            Data:
            {elm.df}
            Normal Estimates: """ + ''.join(
                ["\n\t\t" + str(s_t) + ": " + (str(estimates[s_t]) if s_t in estimates else "n/a") for s_t in
                 schedule]) + "\n\tSlope Ranges:" + ''.join(
                ["\n\t\t" + str(schedule[i]) + " - " + str(schedule[i + 1]) + ": " + str(e) for i, e in
                 enumerate(elm.get_slope_ranges())]) + f"""
            Last size: {last_size}
            Optimistic offset at last evaluated anchor {last_size}: {last_conf[0]}
            Optimistic slope from last segment: {optimistic_slope}
            Remaining steps: {(target_anchor - last_size)}
            Most optimistic value possible at target size {target_anchor}: {optimistic_slope * (target_anchor - last_size) + last_conf[0]}""")
            return np.nan, normal_estimates_last["mean"], estimates, elm

        elif not enforce_all_anchor_evaluations and (elm.get_mean_performance_at_anchor(s_t) < r or (
                t >= 3 and elm.get_lc_estimate_at_target(
                target_anchor) <= r + MAX_ESTIMATE_MARGIN_FOR_FULL_EVALUATION)):
            t = T
            if (elm.get_mean_performance_at_anchor(s_t) < r):
                logger.info(
                    f"Current mean is {elm.get_mean_performance_at_anchor(s_t)}, which is already an improvement over r = {r}. Hence, stepping to full size.")
            else:
                logger.info(
                    f"Candidate appears to be competitive (predicted performance at {target_anchor} is {elm.get_lc_estimate_at_target(target_anchor)}. Jumping to last anchor in schedule: {t}")
        else:
            t += 1
            logger.info(
                f"Finished schedule on {s_t}, and t is now {t}. Performance: {elm.get_normal_estimates(s_t, 4)}.")
            if t < T:
                estimates = elm.get_normal_estimates()
                logger.debug("LC: " + ''.join(["\n\t" + str(s_t) + ": " + (
                    str(estimates[s_t]) if s_t in estimates else "n/a") + ". Avg. runtime: " + str(
                    np.round(np.mean(elm.get_runtimes_at_anchor(s_t) / 1000), 1)) for s_t in schedule]))
                if t > 2:
                    logger.debug(
                        f"Estimate for target size {target_anchor}: {elm.get_performance_interval_at_target(target_anchor)[1]}")

    # output final reports
    toc = time.time()
    estimates = elm.get_normal_estimates()

    # return result depending on observations and configuration
    if len(estimates) == 0 or elm.get_best_worst_train_score() > r:
        logger.info(f"Observed no result or a train performance that is worse than r. In either case, returning nan.")
        return np.nan, np.nan, dict(), elm
    elif len(estimates) < 3:
        max_anchor = max([int(k) for k in estimates])
        return estimates[max_anchor]["mean"], estimates[max_anchor]["mean"], estimates, elm
    else:
        max_anchor = max([int(k) for k in estimates])
        target_performance = estimates[max_anchor][
            "mean"] if t == T or not return_estimate_on_incomplete_runs else elm.get_lc_estimate_at_target(
            target_anchor)
        logger.info(f"Target performance: {target_performance}")
        return target_performance, estimates[max_anchor]["mean"], estimates, elm
