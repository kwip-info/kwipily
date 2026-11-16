"""Static synthetic v1 states for historical readers; deliberately no legacy fitter."""
import copy
import json
from pathlib import Path
from runner import dataflow as df


def frozen_state(table,operator,configuration,graph_digest):
    key=df.digest({'operator':operator,'configuration':configuration,'graphDigest':graph_digest,'columns':table.columns,'rows':list(table.rows),'rowIDs':list(table.row_ids),'sourceDigest':table.source_digest,'role':table.role,'partitionDigest':table.partition_digest,'fittedStates':list(table.fitted_states)})
    fixtures=json.loads(Path(__file__).with_name('fixtures').joinpath('retired-fitted-states-v1.json').read_text())
    if key not in fixtures['states']:raise AssertionError('This historical fixture was not captured; use current Preprocess for new learning tests.')
    return copy.deepcopy(fixtures['states'][key])
