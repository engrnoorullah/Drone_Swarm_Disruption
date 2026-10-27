# User guide

## Environment

Create a virtual environment and install `requirements.txt` from the repository root. Dependencies are NumPy, NetworkX, Matplotlib, SciPy and ImageIO. The example does not require TensorFlow, PyTorch or PyTorch Geometric. `requirements-tested.txt` records the direct versions used for local checks; it is not a complete transitive lockfile.

## Command-line example

```sh
python run_demo.py --nodes 27 --steps 3 --seed 10 --method degree --output outputs/example
```

| Argument | Default | Meaning |
|---|---|---|
| `--nodes` | 27 | Initial node count, 2-1000 |
| `--steps` | 3 | Number of removals, zero to initial node count |
| `--seed` | 10 | Seed for Python random and NumPy |
| `--method` | degree | degree, betweenness, closeness, weight or greedy |
| `--output` | outputs/demo | Directory for the three output files |

Use different output directories for comparisons; the three output files are overwritten in an existing destination. The example runs in one process with Matplotlib's non-interactive Agg backend. Greedy examines every remaining candidate, and scoring repeatedly calculates shortest paths. Large node counts can be expensive; the argument limit is not a promise of fast execution.

## Sequence and interpretation

The initial graph is created after setting both random seeds and scored at step zero. Each subsequent step selects a node, removes it, moves the survivors towards the target and records the score. Ties are resolved by the first node in sorted order.

Greedy uses independent deep copies for trial scoring. It selects the candidate with the lowest resulting shared score at the current positions. Trial scoring does not advance movement or modify the parent graph's `Theta`. This defined example is separate from the original two-score greedy driver.

The default model uses the repository's small-world-style spatial construction, target `(9, 9, 4)` and range `1`. Positions form a compact grid. `update_positions()` translates the centroid towards the target and clips coordinates to the bounds. It does not rebuild edges after movement. The example retains initial edges minus removed-node edges; it does not simulate a fully recomputed dynamic radio mesh.

## Outputs

`metrics.csv` contains `step`, `removed_node`, `remaining_nodes`, `edges`, `largest_component`, `score`, `score_ratio` and `component_factor`. The initial removed-node entry is empty. Score ratio divides by the initial score, or is empty when that score is zero.

The shared score combines accumulated `Theta` values with a normalised sum of hop-distance reliability over unordered node pairs. Scoring increases `Theta` using target proximity, so the score or ratio need not decrease at every step and may exceed one. The pair denominator and initial weight remain those of the initial graph. Empty graphs produce zero.

`config.json` records versions, arguments and movement scope. `score.png` plots the raw score, not a paper figure or confidence interval. Compare matched seeds and configurations. Multiple seeds and a justified statistical analysis are necessary for research conclusions.

## Original drivers

`Drone_Swarms_Code.py` compares betweenness, degree, closeness, weight, degree/weight and DQN through `Removal_Methods.py`. Its default includes DQN and cannot complete that method without the missing checkpoint. For non-DQN investigation, remove `dqn` from the `methods` list in your working copy. It creates separate random graphs for method calls; a paired comparison needs matched initial states.

`Greedy_Heuristic_Code.py` uses its own two score functions and writes Cs/COOP snapshots and CSV files. It is a research driver with separate defaults, not another interface to the demo.

`Power_of_attck_Optimal_Check.py` enumerates four-node removals. Its default 25-node graph means 12,650 combinations. Consider the combinatorial cost before scaling up. Enumeration alone does not establish that its objective and movement sequence match a particular paper experiment.

The drivers write beneath `Custom_Path/` relative to the working directory. Their defaults are in the source, and they do not accept the demo's CLI arguments. Alternative graph classes are not automatically selected by a demo argument.

## Troubleshooting

- Module missing: activate the same environment used for installation and use `python -m pip`.
- Headless plotting: use `MPLBACKEND=Agg` for original scripts on Linux/macOS. The demo selects Agg itself.
- Checkpoint missing: use the core example. Installing optional requirements does not supply the DQN weights.
- Unexpected score differences: record the commit and seed and read the changelog. Original and corrected scores are not interchangeable.
- Slow run or excessive files: reduce nodes and steps, use the single-process example and an isolated output directory.
