from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.preprocessings.transformpipeline import TransformPipeline
from imfas.data.preprocessings.table_transforms import *

class AlgorithmMetaFeatures(DatasetMetaFeatures):
    def __init__(self, path, transforms: TransformPipeline = None, *args, **kwargs):
        super(AlgorithmMetaFeatures, self).__init__(path, transforms, *args, **kwargs)


if __name__ == "__main__":
    # TODO Make exactly this config into a config file!
    path = "/home/ruhkopf/PycharmProjects/AlgoSelectionMF/data/raw/LCBench/config.csv"
    metafeatures = AlgorithmMetaFeatures(path, index_col=0)

    # check if a single transform works as expected
    df = TransformPipeline([Scalar(columns=["batch_size", "max_dropout"])]).fit(metafeatures.df)
    print(df[["batch_size", "max_dropout"]])

    # ensure to tensor will work (dropping categoricals & convert 'num_layers'
    categoricals = [
        "imputation_strategy",
        "learning_rate_scheduler",
        "loss",
        "network",
        "normalization_strategy",
        "optimizer",
        "activation",
        "mlp_shape",
    ]

    pipe = TransformPipeline(
        [
            Scalar(columns=["batch_size", "max_dropout"]),
            Drop(categoricals),
            Replace(columns=["num_layers"], replacedict={"True": 1}),
            Convert(columns=["num_layers"], dtype="numeric"),
            ToTensor(),
        ]
    )
    tensor = pipe.fit(metafeatures.df)
    # check that reapplying a fitted pipe works
    pipe.transform(df)
    print(tensor)

    metafeatures = AlgorithmMetaFeatures(path, transforms=pipe, index_col=0)

    # indexing is rowmajor only on tensors!
    metafeatures[1]
