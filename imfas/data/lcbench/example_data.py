"""
Convenience file to generate some example pipeline on LCBench to import for dev purposes.
"""
from pathlib import Path

import imfas.data.preprocessings as prep
from imfas.data import Dataset_Join_Dmajor, Dataset_LC, DatasetMetaFeatures
from imfas.utils.traintestsplit import train_test_split

train_split, test_split = train_test_split(n=35, share=0.8)

root = Path(__file__).parents[3]

dataset_name = "LCBench"
data_path = root / 'data' / 'raw' / dataset_name
pipe_lc = prep.TransformPipeline(
    [prep.Column_Mean(), prep.Convert(),
     prep.LC_TimeSlices(slices=list(range(51)))]
)

pipe_meta = prep.TransformPipeline(
    [prep.Zero_fill(), prep.Convert(), prep.ToTensor(), prep.ScaleStd()]
)

pipe_algo = prep.TransformPipeline(
    [prep.Zero_fill(),
     prep.Drop(
         ['imputation_strategy', 'learning_rate_scheduler', 'loss', 'network',
          'normalization_strategy', 'optimizer', 'activation', 'mlp_shape', ]),
     prep.Replace(columns=['num_layers'], replacedict={'True': 1}),
     prep.Convert(),
     prep.ToTensor(),
     prep.ScaleStd()]
)

test_dataset = Dataset_Join_Dmajor(
    meta_dataset=DatasetMetaFeatures(
        path=data_path / 'meta_features.csv',
        transforms=pipe_meta),
    learning_curves=Dataset_LC(
        path=data_path / 'logs_subset.h5',
        transforms=pipe_lc,
        metric='Train/train_accuracy'),
    split=test_split,
)

# Show, that we can also input another model and train it beforehand.
train_dataset = Dataset_Join_Dmajor(
    meta_dataset=DatasetMetaFeatures(
        path=data_path / 'meta_features.csv',
        transforms=pipe_meta),
    learning_curves=Dataset_LC(
        path=data_path / 'logs_subset.h5',
        transforms=pipe_lc,
        metric='Train/train_accuracy'),
    split=train_split,
)
