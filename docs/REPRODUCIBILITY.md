# Reproducibility and limitations

## Checked scope

The maintained example and tests cover the default graph, shared score and five example removal methods. Checks include seeded repeatability, node counts, degree/range constraints, empty graphs, pair normalisation and trial-scoring isolation. They do not reproduce the full article, validate physical countermeasures or establish an optimal disruption policy.

A full paper reproduction needs a mapping from each experiment to its revision, graph variant, constants, seed schedule, analysis and output. That mapping and original result data are not supplied as a complete package here.

## Original-driver limitations

- The default `SwarmGraph` does not rebuild edges after movement. Alternative classes differ. A driver's output label does not select its imported graph class.
- The original comparison driver generates separate random graphs per method. A paired evaluation needs matched states and a documented randomness schedule.
- `COOP_Calc.py` differs from the two score functions in `Greedy_Heuristic_Code.py`. Not every accepted parameter is used. The shared function returns zero for the timing field; that is not a measured runtime.
- Scoring mutates `Theta`. The maintained example deep-copies trials; the original drivers need separate review of state, movement and normalisation during candidate comparison.
- The original driver's exception handling can return incomplete results after printing an error. A CSV file alone does not demonstrate that every method succeeded.
- The original large multi-process and greedy experiments were not run end to end during maintenance. Their outputs are not independently verified by the new tests.

## DQN comparison

The code references `nrange_50_iter_14294.ckpt` beneath `Custom_Path/`; no compatible checkpoint is included. It uses TensorFlow's v1 compatibility API and older layers interfaces. `requirements-dqn.txt` suggests an experimental older TensorFlow range, not a tested environment. Use a separate Python 3.10/3.11 environment if investigating it.

Installation is not the only issue. `DQN_train.py` refers to `Batchgraph` without importing it, uses an undefined `inf` name in an action branch, and defines an environment placeholder. Action indexing and a 50-action checkpoint require reconciliation with graph size and node labels. Random weights must not be substituted and described as a reproduced comparison.

DQN is a comparison component, not the novel contribution claimed by this maintenance update. Core checks do not establish that this legacy path executes successfully.

## Corrections affecting results

October 2026 corrections change generation for some node counts, enforce the stated degree cap at both endpoints and fix the shared pair normalisation. They can alter topology, scores and rankings. Do not assign the original published numbers to the corrected implementation without rerunning and validating the relevant experiments.

The unmodified upload is at commit `a17c3e5445267e35e607e5c43ad23a52847b2dc0`. Cite a maintained run's current commit and configuration separately.
