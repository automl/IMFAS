from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa


class AlgoSurrogateLookup(BaseEstimator):
    """sklearn estimator API for multi-fidelity lookup in HalvingGridSearchCV,
    an instance of this class serves as an algo surrogate. This way, HalvingGridSearchCV
    can spawn the algorithms at the current budget and lookup their performance."""

    def __init__(self, slices, algo_id=None, budget=None):
        """
        :param algo_id: algorithms ids available in slices (relevant for fidelity progressions
        from HalvingGridSearchCV.
        :param slices: data.Dataset_Join_Dmajor object, that holds the data of the learning curve slices.
        :param budget: current instantiation's lookup table - required for HalvingGridSearchCV
        """
        self.algo_id = algo_id
        self.slices = slices
        self.budget = budget

    def score(self, X, y):
        # ignore X, y, since we are not fitting the algo, but lookup it's performacne

        # 0 index is appropriate only if successively halving is used and the split variable is
        # overriden at each iteration. so in split always the dataset's id of the test dataset is
        # written to
        # 1 index referes to the tuple from dataset_Dmajor's get_item'r return value
        return self.slices[0][1][int(self.budget)][self.algo_id].numpy()

    def set_param(self, budget, algo_id):
        """Set the param to the current budget, returning an instance of the
        lookup-estimator for the respective budget"""
        self.budget = budget
        self.algo_id = algo_id

        return self

    def fit(self, X, y):
        return self

# slices must be the entire range of available budgets - alternatively, we could
# make use of yahpo's surrogate- budget lookup directly. - which is more flexible
# search = HalvingGridSearchCV(AlgoSurrogateLookup(slices), param_grid, resource='budget',
#                              max_resources=10,
#                              random_state=0).fit(X, y)
# search.best_params_
#
# search.best_estimator_.estimators_
