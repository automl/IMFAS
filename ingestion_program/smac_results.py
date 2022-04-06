import os
import json
import pandas as pd
from itertools import chain
from ingestion_program.ingestion_smac import tae, kf, env  # just for explicitness

run_id = '1608637542_vae'

root_dir = '/'.join(os.getcwd().split('/')[:-1])  # fixing the root to project root and not ingestion_program
with open(f'{root_dir}/output/run_{run_id}/runhistory.json') as json_file:
    runhistory = json.load(json_file)

# preprocess config dict
configs = runhistory['configs']
config_df = pd.DataFrame.from_dict(configs, orient='index')

# preprocess the rundata dict
rundata = [list(chain(*i)) for i in runhistory['data']]
run_df = pd.DataFrame([[list(item.values()) if isinstance(item, dict) else item
                        for item in step] for step in rundata],
                      columns=(
                          'config_id', 'instance_id', 'seed', 'budget', 'cost', 'time', 'status', 'starttime',
                          'endtime',
                          'additional_info'))
run_df['status'] = run_df['status'].apply(lambda x: x[0])

# add the configs in the df
config_df.index = config_df.index.astype(int)
run_config_df = pd.merge(config_df, run_df, left_index=True, right_on='config_id')

# Find the not failed configs on highest fidelity.
budget_success = (run_df['budget'] == 10000) & (run_df['status'] == 'StatusType.SUCCESS')
final_fidelity = run_config_df[budget_success]
final_fidelity = final_fidelity[['cost', 'budget', 'status', *config_df.columns]].sort_values('cost')

# Write out frame to csv.
# run_config_df.to_csv(f'{root_dir}/output/run_{run_id}/run_{run_id}.csv')
config_names = ['embedding_dim', 'learning_rate_init', 'lossweight1',
                'lossweight2', 'lossweight3', 'lossweight4', 'n_compettitors',
                'repellent_share']

# index is runid!
rerun_config = final_fidelity[config_names].transpose().to_dict()

# single config rerun of TAE:
# TODO create an ID, with which the models are plotted and saved.
cfg = configs['1']
tae(cfg, budget=100)
