# ecliseutils

Shared general-purpose research utilities used across multiple projects (e.g.
`KF_RNN`, `event_camera`). This package is the single source of truth for code
that previously lived as diverged copies inside each project's
`infrastructure/utils.py`.

## What's here

| Module | Contents |
|--------|----------|
| `settings` | `configure(device, dtype, precision, seed, debug)`, `default_dtype`, safe-global registration. No hardcoded paths; no import-time torch mutation. |
| `types` | `ModelPair = tuple[nn.Module, TensorDict]` and shared aliases. |
| `labeled_array` | `LabeledArray` / `LabeledDataset` (in-house replacement for `dimarray`). |
| `modules` | Module/tensor/TensorDict batching: `stack_module_arr`, `run_module_arr`, `multi_vmap`, `td_items`, ... |
| `arrays` | NumPy object-array comprehensions: `multi_map`, `multi_zip`, `broadcast_dim_arrays`, ... |
| `linalg` | `pow_series`, `batch_trace`, `kl_div`, `sqrtm`, `hadamard_conjugation*`, `inverse`, `eig_some`, ... |
| `are` | `solve_discrete_are` (torch Schur) and `solve_continuous_are` (scipy `ordqz` fallback) + residual tests. |
| `ode` | `batch_odeint`, `linspace`, `geomspace` (lazy `torchdiffeq`). |
| `ensemble` | `EnsembleModule` chunked vmap wrapper. |
| `fast_conv_scan` | `conv_scan` exponential cumulative scan autograd op. |
| `recursive` | Dotted-path `rgetattr` / `rsetattr` / `rgetitem` / `rsetitem`. |
| `dicts` | `flatten_nested_dict`, `map_dict`, `nested_type`, `call_func_with_kwargs`. |
| `io` | `torch_load`, `empty_cache`, `reset_seed`, `model_size`. |
| `memory` | `get_tensors_in_memory`, `track_tensor_diff`, ... |
| `timing` | `Timer`, `print_disabled`, `print_enabled`, `track_calls`. |
| `plotting` | `color`, `confidence_ellipse`. |

## Install

From PyPI (once published):

```bash
pip install ecliseutils
# with ODE support:
pip install "ecliseutils[ode]"
```

From GitHub (works on any device, no clone needed):

```bash
pip install "git+https://github.com/JophiArcana/ecliseutils.git"
# pin to a release:
pip install "git+https://github.com/JophiArcana/ecliseutils.git@v0.1.0"
# with ODE support:
pip install "ecliseutils[ode] @ git+https://github.com/JophiArcana/ecliseutils.git"
```

Editable (for local development):

```bash
pip install -e /path/to/ecliseutils
# with ODE support:
pip install -e "/path/to/ecliseutils[ode]"
```

## Usage

```python
import torch
import ecliseutils as eu

# Configure once at startup (typically from your project's settings module).
eu.configure(device="cpu", dtype=torch.float64, precision=8, seed=1212)

P = eu.solve_discrete_are(A, B, Q, R)
```

`torchdiffeq` is only required if you call `ecliseutils.ode.batch_odeint`; it is
imported lazily.
