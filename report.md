| Path                                             | Metric                 | main   | workspace   | Change   |
|--------------------------------------------------|------------------------|--------|-------------|----------|
| pipelines/tf_cnn/metrics/scores.json             | accuracy               | -      | 0.6105      | -        |
| pipelines/tf_cnn/metrics/scores.json             | adjusted_rand_score    | -      | 0.34023     | -        |
| pipelines/tf_cnn/metrics/scores.json             | balanced_accuracy      | -      | 0.61579     | -        |
| pipelines/sklearn_classifier/metrics/scores.json | accuracy               | -      | 0.96        | -        |
| pipelines/sklearn_classifier/metrics/scores.json | adjusted_rand_score    | -      | 0.89179     | -        |
| pipelines/sklearn_classifier/metrics/scores.json | balanced_accuracy      | -      | 0.95694     | -        |
| pipelines/sklearn_regression/metrics/scores.json | neg_mean_squared_error | -      | -2798.30884 | -        |
| pipelines/sklearn_regression/metrics/scores.json | r2                     | -      | 0.51378     | -        |

usage: dvc metrics [-h] [-q | -v] {show,diff} ...

Commands to display and compare metrics.
Documentation: <https://man.dvc.org/metrics>

positional arguments:
  {show,diff}    Use `dvc metrics CMD --help` to display command-specific
                 help.
    show         Print metrics, with optional formatting.
    diff         Show changes in metrics between commits in the DVC
                 repository, or between a commit and the workspace.

optional arguments:
  -h, --help     show this help message and exit
  -q, --quiet    Be quiet.
  -v, --verbose  Be verbose.
