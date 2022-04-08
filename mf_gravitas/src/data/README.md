# Data Preprocessing

Ensembling is used to subselect suitable candidate algorithms (for instance in the HP-Space, if only
a single algo is selected from), that are sufficiently diverse. Since autosklearn's backbone
ensemble algorithm; Caruana, Rich, Alexandru Niculescu-Mizil, Geoff Crew, and Alex Ksikes. 2004.
“Ensemble Selection from Libraries of Models.” In Proceedings of the Twenty-First International
Conference on Machine Learning, 18. ICML ’04. New York, NY, USA: Association for Computing
Machinery. Requires to have access to the prediction of the model (which at least LCBench doesn't
provide), there is another forward selection ensembling method.

## LCBench

metafeatures: fairly many have nan values simply due to memory /time constraints in the landmarking
features.

data_2k & data_2k_lw are the same - only difference: lw has additional features such as gradient
inforamtion.

## HPOBench
