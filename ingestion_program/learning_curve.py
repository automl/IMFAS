import numpy as np
import json

# === Verbose mode
verbose = True


def vprint(mode, t):
    """
    Print to stdout, only if in verbose mode.

    Parameters
    ----------
    mode : bool
        True if the verbose mode is on, False otherwise.

    Examples
    --------
    >>> vprint(True, "hello world")
    hello world

    >>> vprint(False, "hello world")

    """

    if mode:
        print(str(t))


class Learning_Curve:
    """
    A learning curve of an algorithm on a dataset
    """

    def __init__(self, file_path):
        """
        Initialize the learning curve

        Parameters
        ----------
        file_path : str
            Path to the file containing data of the learning_curve

        """
        self.file_path = file_path
        self.scores, self.timestamps = self.load_data()

    def load_data(self):
        """
        Load timestamps and scores from the given path to build a learning curve

        Parameters
        ----------
        file_path : str
            Path to the file containing data of the learning_curve

        Returns
        ----------
        scores : list of str
            List of performance scores
        timestamps : list of float
            List of timestamps associated with the scores

        Examples
        ----------
        >>> lc.load_data()
        scores = [0.73 0.78 ... 0.81 0.81]
        timestamps = [0.62 1.9 ...8 131.8 263.06]

        """
        # vprint(verbose, "file_path = " + self.file_path)

        scores, timestamps = [], []
        try:
            with open(self.file_path, "r") as data:
                lines = data.readlines()
                dictionary = {line.split(":")[0]: line.split(":")[1] for line in lines}
                timestamps = np.around(json.loads(dictionary["times"]), decimals=2)
                scores = np.around(json.loads(dictionary["scores"]), decimals=2)

        # If the data is missing, set timestamp = 0 and score = 0 as default
        except FileNotFoundError:
            scores.append(0.0)
            timestamps.append(0.0)
            dataset_name = self.file_path.split("/")[7]
            algo_name = self.file_path.split("/")[8]
            vprint(
                verbose,
                '*Warning* Learning curve of algorithm "{}" on dataset "{}" is missing, replaced by 0 as default!'.format(
                    algo_name, dataset_name
                ),
            )

        # vprint(verbose, "timestamps = " + str(timestamps))
        # vprint(verbose, "scores = " + str(scores))

        return scores, timestamps

    def get_last_point_within_delta_t(self, delta_t, C):
        """
        Return the last achievable point on the learning curve given the allocated time budget delta_t

        Parameters
        ----------

        delta_t : float
            Allocated time budget given by the agent.
        C : float
            The timestamp of the last point on the learning curve (x-coordinate of current position on the learning curve)

        Returns
        ----------
        score : float
            The last achievable score within delta_t
        timestamp : float
            The timestamp associated with the last achievable score

        Examples
        ----------
        >>> lc.get_last_point_within_delta_t(50, 151.73)
        score = 0.5
        timestamp =  151.73

        """

        temp_time = C + delta_t

        for i in range(len(self.timestamps)):
            if temp_time < self.timestamps[i]:
                if (
                    i == 0
                ):  # if delta_t is not enough to get the first point, the agent wasted it for nothing!
                    score, timestamp = 0.0, 0.0
                else:  # return the last achievable point
                    score, timestamp = self.scores[i - 1], self.timestamps[i - 1]
                return score, timestamp

        # If the last point on the learning curve is already reached, return it
        score, timestamp = self.scores[-1], self.timestamps[-1]
        return score, timestamp
