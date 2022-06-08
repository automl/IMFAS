import pandas as pd


class LCBench_API:
    def __init__(self, json):
        self.data = json
        self.names_datasets = list(self.data.keys())
        self.tags = self.get_queriable_tags(dataset_name="APSFailure")

        self.config, self.logs, self.results = self.parse()

    def query(self, dataset_name, tag, config_id):
        """
        Query a run.

        Keyword arguments:
        dataset_name -- str, the name of the dataset in the benchmark
        tag -- str, the tag you want to query
        config_id -- int, an identifier for which run you want to query, if too large will query the last run
        """
        config_id = str(config_id)
        if dataset_name not in self.names_datasets:
            raise ValueError("Dataset name not found.")

        if config_id not in self.data[dataset_name].keys():
            raise ValueError("Config nr %s not found for dataset %s." % (config_id, dataset_name))

        if tag in self.data[dataset_name][config_id]["log"].keys():
            return self.data[dataset_name][config_id]["log"][tag]

        if tag in self.data[dataset_name][config_id]["results"].keys():
            return self.data[dataset_name][config_id]["results"][tag]

        if tag in self.data[dataset_name][config_id]["config_raw"].keys():
            return self.data[dataset_name][config_id]["config_raw"][tag]

        if tag == "config_raw":
            return self.data[dataset_name][config_id]["config_raw"]

        raise ValueError(
            "Tag %s not found for config_raw %s for dataset %s" % (tag, config_id, dataset_name)
        )

    def get_queriable_tags(self, dataset_name=None, config_id=None):
        """Returns a list of all queriable tags"""
        if dataset_name is None or config_id is None:
            dataset_name = list(self.data.keys())[0]
            config_id = list(self.data[dataset_name].keys())[0]
        else:
            config_id = str(config_id)
        log_tags = list(self.data[dataset_name][config_id]["log"].keys())
        result_tags = list(self.data[dataset_name][config_id]["results"].keys())
        config_tags = list(self.data[dataset_name][config_id]["config"].keys())
        return log_tags + result_tags + config_tags

    def parse(self):
        print()
        data_algo = {(d, a): data for d, algos in self.data.items() for a, data in algos.items()}
        names_algos = self.data[self.names_datasets[0]].keys()

        # logs = {(d, a): v for (d, a), v in data_algo.items()}
        logs = {
            (d, a, logged): v
            for (d, a), data in data_algo.items()
            for logged, v in data["log"].items()
        }

        configs = {k: data["config"] for k, data in data_algo.items()}
        results = {k: data["results"] for k, data in data_algo.items()}

        # parse as multi_index
        logs = pd.DataFrame.from_dict(logs)
        logs.columns.names = ["dataset", "algorithm", "logged"]
        # change the default view for convenience
        logs = logs.T
        logs = logs.reorder_levels(["logged", "dataset", "algorithm"])

        # logs.loc['time']  # conveniently select the tracked feature
        # logged, datasets, algos = logs.index.levels # conveniently get the available options

        # Fixme: make this a debug flag and raise on difference in slices!
        if False:
            # to validate that across datasets the config is always the same
            # --> we only need one config per algorithm!
            config = pd.DataFrame.from_dict(configs)
            config.columns.names = ["dataset", "algorithm"]
            config.T.xs("1", level="algorithm")
            config.T.xs("2", level="algorithm")
            # config.T.xs(0, level='algorithm') # to extract a single algorithm

        config = {a: configs[(self.names_datasets[0], a)] for a in names_algos}
        config = pd.DataFrame.from_dict(config, orient="index")
        config.index.name = "algorithm"

        results = pd.DataFrame.from_dict(results)
        results.columns.names = ["dataset", "algorithm"]
        results = results.T

        return config, logs, results
