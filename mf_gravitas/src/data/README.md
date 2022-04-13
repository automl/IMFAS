# Data Preprocessing

To make the preprocessing end to end reproducible & configurable, pipe_raw.py is a hydra file, that
creates the subsets

Ensembling is used to subselect suitable candidate algorithms (for instance in the HP-Space, if only
a single algo is selected from), that are sufficiently diverse. Since autosklearn's backbone
ensemble algorithm;

> “Ensemble Selection from Libraries of Models.”
> In Proceedings of the Twenty-First International
> Conference on Machine Learning, 18. ICML ’04. New York, NY, USA: Association for Computing
> Machinery
>
> -- <cite>Caruana, Rich, Alexandru Niculescu-Mizil, Geoff Crew, and Alex Ksikes. 2004.</cite>

requires to have access to the prediction of the model (which at least LCBench doesn't provide),
there is another forward selection ensembling method (naive: topk), which work on final
performances.

## LCBench

To download and subset the LCBench dataset to your likings, configure /mf_gravitas/src/data/raw.yaml
and call pipe_raw.py afterwards - to let hydra do the dirty work for you. After calling pipe_raw.py,
Algo/SelectionMF/data will have the following structure:

```
/preprocessed
    
/raw
    /LCBench
        config.csv          # algo_meta_features
        logs.h5             # learning curves
        logs_subset.h5      # subset of learning_curves
        results.h5          # scalar metric (such as final performances)
        results_subset.h5
    
/downloads
    # downloaded json/csv files. 
    data_2k.json, 
    data_2k_lw.json,
    meta_features.json 
```

metafeatures: fairly many have nan values simply due to memory /time constraints in the landmarking
features.

data_2k & data_2k_lw are the same - only difference: lw has additional features such as gradient
inforamtion.
