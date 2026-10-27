# Validation record

Local checks on 27 October 2026 used Python 3.12.14, NumPy 2.3.5, NetworkX 3.4.2, Matplotlib 3.10.8, SciPy 1.17.0 and ImageIO 2.37.0.

```sh
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Ten tests passed. They cover five demo methods and repeatability, requested node counts of 2, 28, 50 and 101, degree and initial distance limits, empty-graph movement, a hand-computed two-node score, zero-weight/disconnected cases, greedy trial isolation, independent exhaustive-worker state, removing every node, invalid arguments and non-DQN imports without TensorFlow.

The CLI was also exercised to generate CSV, configuration and plot output. Generated files are ignored by Git. The original large drivers and checkpoint-based DQN path were not validated end to end. See the reproducibility guide for the distinction between software checks and paper reproduction.
