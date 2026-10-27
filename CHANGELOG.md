# Changes

## 27 October 2026

- Added bounded, seeded `run_demo.py` with five removal methods and CSV, JSON and plot outputs.
- Added setup requirements, tested versions, tests, citation metadata, user guide and reproducibility notes.
- Removed unused PyTorch/PyG imports and moved TensorFlow imports into the DQN-specific method.
- Fixed rounded grid size, which could underfill some node counts. Minimum-separation failures now raise an error.
- Enforced the default graph's degree cap at both edge endpoints and checked rewires before removing an edge.
- Made empty-graph movement a no-op.
- Corrected shared-score pair normalisation to `n * (n - 1) / 2`, using the actual initial size. Previously it used `(NUM_DRONES * NUM_DRONES - 1) / 2` with a global count of 101.
- Handled empty, singleton and zero-initial-weight scoring without division by zero.
- Fixed original greedy-driver checks that could reject selected node ID zero.
- Deep-copied each exhaustive worker's mutable swarm state.

Graph and score corrections can change scientific outputs. This update does not claim a full article rerun. The original upload remains at `a17c3e5445267e35e607e5c43ad23a52847b2dc0` in Git history.
