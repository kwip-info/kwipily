from runner.tests.cli_discovery import capabilities_cli
import json
import subprocess
import sys


def test_cli_reports_exact_layer_norm_execution_domain():
    result = capabilities_cli()
    assert json.loads(result.stdout)['componentCapabilities']['layerNorm'] == {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.layer_norm',
        'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True,
        'epsilonMinimum': 1.1754943508222875e-38, 'epsilonMaximum': 3.4028234663852886e38,
        'elementwiseAffineModes': [False, True], 'biasModes': [False, True],
        'varianceCorrection': 0, 'usesCurrentInputStatistics': True,
    }
