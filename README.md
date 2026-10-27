# Uncertainty-aware drone swarm disruption

Python research software for modelling UAV communication graphs, evaluating cooperative scores and studying node-removal strategies. Associated research:

Noor Ullah, Youcef Djenouri, Tomasz Michalak, Ahmed Nabil Belbachir and Gautam Srivastava. **Uncertainty-Aware Drone Swarm Disruption for Threat Mitigation.** *IEEE Internet of Things Journal*, 13(2), 1978-1987, 2026. [DOI: 10.1109/JIOT.2025.3626254](https://doi.org/10.1109/JIOT.2025.3626254). First published online in October 2025.

The repository includes the original experiment drivers and a small command-line example added during maintenance in October 2026. The example is tested and runs without TensorFlow or PyTorch. It demonstrates the graph and scoring code; it is **not a reproduction of the article's experiments**. See [reproducibility and limitations](docs/REPRODUCIBILITY.md) before interpreting results.

## Quick start

Use Python 3.10 or later. Python 3.12 was used for the recorded checks.

```sh
git clone https://github.com/engrnoorullah/Drone_Swarm_Disruption.git
cd Drone_Swarm_Disruption
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py --nodes 27 --steps 3 --seed 10 --method degree --output outputs/degree
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. If your interpreter is named `python3`, use that when creating the environment. No GPU or trained model is needed for the example.

The command writes `metrics.csv` with an initial row and each removal step, `config.json` with arguments and package versions, and `score.png` with a plot of the recorded repository score.

```sh
python run_demo.py --nodes 27 --steps 3 --seed 10 --method greedy --output outputs/greedy
python -m unittest discover -s tests -v
```

Supported example methods are `degree`, `betweenness`, `closeness`, `weight` and `greedy`. Greedy scores each candidate on a separate copy of the current graph and is slower than a degree-based choice. See [the user guide](docs/USER_GUIDE.md) for arguments and output definitions.

## Repository map

| File | Role |
|---|---|
| `run_demo.py` | Bounded, seeded command-line example with CSV, JSON and plot output |
| `SwarmGraph.py` | Default spatial graph and centroid-based movement |
| `SwarmGraph_ER.py`, `Swarm_Graph_BA.py`, `SwarmGraph_PSO.py` | Alternative experimental graph and movement implementations |
| `COOP_Calc.py` | Shared cooperative score and proximity updates |
| `Drone_Swarms_Code.py`, `Removal_Methods.py` | Original comparison driver and removal strategies |
| `Greedy_Heuristic_Code.py` | Original greedy experiment driver with two scoring variants |
| `Power_of_attck_Optimal_Check.py` | Original four-node enumeration experiment |
| `DQN_train.py`, `dqn_batch.py`, `batch.py`, `uavs_env.py` | Legacy TensorFlow comparison components |
| `utilities.py` | Research plotting and CSV utilities |
| `tests/` | Checks of graph generation, scores, imports and the example |

## Original experiments and validation

The original drivers contain source-level configuration and use a relative directory named `Custom_Path`. They may create many images and use all CPU cores. Start with the small example before attempting them.

The DQN comparison needs a compatible checkpoint that is **not included**. Additional legacy-code issues are recorded in [the reproducibility guide](docs/REPRODUCIBILITY.md). Installing TensorFlow alone does not make that comparison reproducible. Non-DQN modules can now be imported without deep-learning packages.

[TESTING.md](docs/TESTING.md) records the local checks. [CHANGELOG.md](CHANGELOG.md) identifies corrections that can change generated graphs or scores. The original upload is available at commit `a17c3e5445267e35e607e5c43ad23a52847b2dc0`. Software tests do not independently validate the article's scientific conclusions.

## Citation and reuse

Please cite the article for the associated research, and record the commit, configuration and environment for software results. `CITATION.cff` provides citation metadata.

No redistribution licence has been selected for this repository. Public availability alone does not grant an open-source licence. The ongoing IIT-CNR XR software and datasets are separate from this public repository.
