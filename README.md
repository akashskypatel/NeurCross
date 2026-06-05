# NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation

### [Project](https://qiujiedong.github.io/publications/NeurCross/) | [Paper](https://arxiv.org/pdf/2405.13745)

**This repository is the official PyTorch implementation of our
paper,  *NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation*, ACM Transactions on Graphics (SIGGRAPH 2025).**

<img src='./assets/NeurCross.jpg'>

## Requirements

- python 3.7
- CUDA 11.7
- Pytorch 1.13.1

## Installation

```
git clone https://github.com/QiujieDong/NeurCross.git
cd NeurCross
```

## Build A Wheel

Install the build frontend, then create a wheel from the repository root:

```powershell
python -m pip install --upgrade build
python -m build --wheel
```

The generated wheel is written to `dist/`. The default wheel is a pure Python wheel, for example `neurcross-0.1.0-py3-none-any.whl`.

On Windows, you can use the helper script:

```powershell
.\build-wheel.ps1
```

To append optional Torch and CUDA build metadata to the wheel filename, use:

```powershell
.\build-wheel.ps1 -IncludeTorchVersion -IncludeCudaVersion
```

That produces a wheel name with a standard wheel build tag, for example:

```text
neurcross-0.1.0-1torch2120cu132-py3-none-any.whl
```

To install the built wheel:

```powershell
python -m pip install .\dist\neurcross-0.1.0-py3-none-any.whl
```

The wheel packages the Python modules and exposes a console entry point:

```powershell
neurcross-train-quad-mesh --data_path D:\path\to\mesh.ply
```

The package also exposes a high-level module entry point:

```powershell
python -m neurcross --help
python -m neurcross train-quad-mesh --help
python -m neurcross crossfield-to-rosy --help
```

It also exposes a conversion helper for downstream QuadWild-compatible `.rosy` files:

```powershell
neurcross-crossfield-to-rosy D:\path\to\crossfield.txt
```

## Using NeurCross From Another Python Module

The installed package exposes a small programmatic API through `neurcross`:

- `neurcross.train_crossfield(...)`
- `neurcross.convert_crossfield_to_rosy(...)`
- `neurcross.convert_crossfield_to_rawfield(...)`

Example:

```python
import neurcross

result = neurcross.train_crossfield(
    data_path=r"D:\path\to\mesh.ply",
    out_dir=r"D:\path\to\output",
    n_samples=1000,
    n_points=1000,
    num_epochs=4,
    fast_nondeterministic=True,
)

print(result.output_dir)
print(result.log_path)
print(result.total_elapsed_seconds)
print(result.stopped_early)

rosy_path = neurcross.convert_crossfield_to_rosy(
    r"D:\path\to\output\mesh\save_crossField\mesh_final.txt"
)

rawfield_path = neurcross.convert_crossfield_to_rawfield(
    r"D:\path\to\output\mesh\save_crossField\mesh_final.txt"
)
```

`train_crossfield(...)` returns a `TrainingResult` object with:

- `args`
- `output_dir`
- `log_path`
- `mesh_name`
- `total_elapsed_seconds`
- `stopped_early`
- `stop_summary`

If you prefer CLI-style argument forwarding from Python, `train_crossfield` also accepts `argv`:

```python
import neurcross

result = neurcross.train_crossfield(
    argv=[
        "--data_path", r"D:\path\to\mesh.ply",
        "--out_dir", r"D:\path\to\output",
        "--n_samples", "1000",
        "--n_points", "1000",
        "--num_epochs", "4",
    ]
)
```

`convert_crossfield_to_rosy(...)` reads a saved NeurCross `.txt` cross-field snapshot and writes a QuadWild-compatible `.rosy` file.

`convert_crossfield_to_rawfield(...)` reads a saved NeurCross `.txt` cross-field snapshot and writes a Directional-compatible `.rawfield` file. This requires the snapshot rows to contain both cross-field branches, so each row must provide at least 6 floating-point values.

The source checkout includes `data/doubleTorus/input/doubleTorus.ply`, so `--data_path` can be omitted when training from the repo. The wheel does not bundle sample training data, so `--data_path` is required after installation.

## Overfitting

```
cd quad_mesh
python train_quad_mesh.py
```

You can also override parameters from the command line:

```powershell
python train_quad_mesh.py --data_path D:\path\to\mesh.ply --n_samples 10000 --lr 5e-5
```

Equivalent installed-package usage:

```powershell
python -m neurcross train-quad-mesh --data_path D:\path\to\mesh.ply --out_dir D:\path\to\output
```

Estimated runtime from the paper: for a triangular mesh with 50,000 faces, each optimization iteration takes about `68.34 ms`, and the default research setting uses `10,000` iterations. That corresponds to roughly `683.4` seconds, or about `11.4` minutes, for one full run under that configuration. Actual runtime in this repository will vary with GPU, PyTorch/CUDA version, mesh complexity, and your chosen `--n_samples` setting.

## `quad_mesh_args.py` Reference

The training entry point accepts the following arguments.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--out_dir` | `./output/` | Output directory used for logs and training artifacts. The script creates a subdirectory named after the input mesh file. |
| `--model_name` | `model` | Model name placeholder for saved artifacts. The current training script keeps it for compatibility with the original project setup. |
| `--seed` | `3627473` | Random seed applied to PyTorch, NumPy, and Python's `random` module for reproducible runs. |
| `--data_path` | repo sample mesh if available, otherwise `None` | Path to the input surface mesh used for training. This must point to a mesh file supported by `trimesh`. It is required when running from an installed wheel. |
| `--n_samples` | `10` | Number of dataset samples exposed per epoch through the dataset length. This directly affects the number of training iterations for each epoch. |
| `--n_points` | `15000` | Number of points sampled per training item for manifold and non-manifold point sets. Larger values increase memory and compute cost. |
| `--grid_res` | `256` | Uniform grid resolution parameter passed into the dataset. It is part of the original training configuration, though the current dataset code does not use it directly. |
| `--nonmnfld_sample_type` | `gaussian` | Intended strategy for off-surface sampling. Accepted values in the help text are `grid`, `gaussian`, and `combined`, but the current dataset implementation always samples uniformly in the configured range. |
| `--num_epochs` | `1` | Number of epochs to run. The original script comment indicates this was expected to stay at `1` for the provided workflow. |
| `--lr` | `5e-5` | Adam learning rate used for optimizing the model. |
| `--grad_clip_norm` | `10.0` | Gradient clipping threshold. Set to `0` or a negative value to disable clipping. |
| `--batch_size` | `1` | Mini-batch size used by the PyTorch `DataLoader`. Larger values require more GPU memory. |
| `--load_path` | `None` | Optional checkpoint path. If provided, the model weights are loaded before training continues. |
| `--num_workers` | `4` | Number of `DataLoader` worker processes used for training batches. |
| `--persistent_workers` | disabled | Keeps `DataLoader` workers alive across epochs to reduce worker startup overhead. |
| `--fast_nondeterministic` | disabled | Allows faster nondeterministic CUDA/cuDNN behavior instead of fully deterministic seeding. |
| `--init_type` | `siren` | Decoder initialization strategy. The help text lists `siren`, `geometric_sine`, `geometric_relu`, and `mfgi`. |
| `--decoder_hidden_dim` | `256` | Width of the decoder hidden layers. |
| `--decoder_n_hidden_layers` | `4` | Number of hidden layers used in the decoder network. |
| `--latent_size` | `0` | Latent code size placeholder. The current quad mesh training path keeps this for compatibility with the original architecture. |
| `--nl` | `sine` | Nonlinearity used by the network, such as `sine`, `relu`, or `softplus`. |
| `--sphere_init_params` | `[1.6, 0.1]` | Parameters controlling sphere-based initialization behavior, interpreted by the model initialization code as radius and scaling values. |
| `--udf` | disabled | Enables unsigned distance field behavior in the model if specified. |
| `--output_any` | disabled | Optional flag preserved from the original project. It toggles alternate output behavior where supported by downstream code. |
| `--loss_type` | `siren_wo_n_w_morse_w_theta` | Selects the configured loss composition used by the quad mesh training pipeline. |
| `--decay_params` | `[3, 0.2, 3, 0.4, 0.001, 0]` | Parameters controlling scheduled decay behavior inside the Morse loss update step. |
| `--morse_type` | `l1` | Norm type used for the Morse divergence term. The help text lists `l1` and `l2`. |
| `--morse_decay` | `linear` | Decay schedule for Morse-loss weighting. Supported values in the help text are `none`, `step`, and `linear`. |
| `--loss_weights` | `[7000.0, 600.0, 10, 50.0, 30, 3]` | Per-term loss weights, documented in code as `sdf`, `inter`, `normal`, `eikonal`, `div`, and `morse`. |
| `--morse_near` | disabled | If enabled, the Morse loss uses the sampled `near_points` in addition to the default point sets. |
| `--weight_for_morse` | disabled | If enabled, reweights the Morse term according to the distance of each sampled point. |
| `--use_morse_nonmnfld_grad` | `True` | Controls whether Morse loss is applied to non-manifold gradients. |
| `--relax_morse` | `0.5` | Upper bound used by the relaxed Morse formulation. |
| `--use_vertices` | `False` | Controls whether to use vertices directly instead of the default sampled points. The code comment suggests `False` is used to avoid overfitting. |
| `--featureLine_threshold` | `1.0` | Threshold related to feature-line behavior in the cross-field pipeline. |
| `--convert_crossfield_to_rosy` | disabled | If enabled, every saved `save_crossField/*.txt` snapshot is also converted into a QuadWild-compatible `.rosy` sidecar file. |
| `--early_stop` | disabled | Enables early stopping based on smoothed loss plateau detection and optional theta-term thresholds. |
| `--early_stop_min_steps` | `1000` | Minimum number of global training steps before early stopping can trigger. |
| `--early_stop_patience` | `500` | Number of steps without sufficient smoothed-loss improvement before plateau stopping triggers. |
| `--early_stop_min_delta` | `1e-3` | Minimum smoothed-loss improvement required to reset early-stop patience. |
| `--early_stop_smooth_window` | `50` | Moving-average window size used for smoothing the total loss for early stopping. |
| `--early_stop_check_interval` | `10` | Evaluate the early-stop controller every `N` global steps. |
| `--early_stop_target_loss` | `None` | Optional smoothed total-loss target that can stop training once the minimum-step guard is satisfied. |
| `--early_stop_theta_neighbor_threshold` | `None` | Optional maximum unweighted theta-neighbor term required before early stopping is allowed. |
| `--early_stop_theta_hessian_threshold` | `None` | Optional maximum unweighted theta-hessian term required before early stopping is allowed. |

Notes:

- Boolean flags defined with `action='store_true'` such as `--udf`, `--output_any`, `--morse_near`, and `--weight_for_morse` are disabled by default and become enabled when the flag is present.
- `--use_morse_nonmnfld_grad` and `--use_vertices` use `type=bool`, so if you pass them explicitly on the command line, use forms such as `--use_vertices True` or `--use_vertices False`.
- Some arguments are preserved from the original research code even when the current training path uses them lightly or not at all. The table above reflects the behavior of the current repository state.

## Early Stopping

NeurCross supports opt-in early stopping based on smoothed total-loss plateau detection. This is intended as a practical time-saving guard, not a replacement for downstream field or remeshing validation.

Example:

```powershell
python -m neurcross train-quad-mesh `
  --data_path D:\path\to\mesh.ply `
  --out_dir D:\path\to\output `
  --num_epochs 20 `
  --early_stop `
  --early_stop_min_steps 2000 `
  --early_stop_patience 1000 `
  --early_stop_smooth_window 100 `
  --early_stop_theta_neighbor_threshold 1e-3 `
  --early_stop_theta_hessian_threshold 1e-3
```

When early stopping triggers, the current field is still exported and marked as the final output.

### Cross-field To `.rosy`

NeurCross saves intermediate cross-field snapshots under:

```text
<out_dir>\<mesh-name>\save_crossField\
```

The current export manager preserves history snapshots and also maintains stable aliases for downstream tools:

```text
save_crossField\
  <mesh-name>_iter_<global_step>.txt
  <mesh-name>_latest.txt
  <mesh-name>_best.txt
  <mesh-name>_final.txt
  <mesh-name>_latest.meta.txt
  <mesh-name>_best.meta.txt
  <mesh-name>_final.meta.txt
```

`latest` is overwritten every export, `best` is updated when the field-quality score improves, and `final` is written when training completes or stops early.

If `--convert_crossfield_to_rosy` is enabled, these `.txt` outputs also receive QuadWild-compatible `.rosy` sidecars during training.

The preserved `*_iter_<global_step>.txt` snapshots use the global training step, so multi-epoch runs no longer overwrite earlier exports.

You can also convert a saved cross-field file manually:

```powershell
# convert to rosy format
neurcross-crossfield-to-rosy D:\path\to\crossfield_iter_499.txt
# convert to rawfield format
python -m quad_mesh.convert_crossfield D:\path\to\crossfield_iter_499.txt --format rawfield
```

## Metrics Reporting

Every preserved cross-field export has a matching JSON metrics report under:

```text
<out_dir>\<mesh-name>\metrics\
```

The metrics directory contains:

```text
metrics\
  <mesh-name>_iter_<global_step>.json
  <mesh-name>_latest.json
  <mesh-name>_best.json
  <mesh-name>_final.json
```

Each report records:

- training losses: total, manifold, non-manifold, eikonal, Morse, theta-Hessian, theta-neighbor
- field-validity metrics: tangency, norm error, orthogonality, handedness, flipped-frame ratio, NaN count
- field-smoothness metrics: adjacent cross-field error mean, median, p95, and max
- singularity proxy metrics
- a composite field score used to rank `best`

The current `best` snapshot is selected using a field-oriented score derived from theta alignment, neighboring smoothness, tangency error, and flipped-frame ratio.

## Cite

If you find our work useful for your research, please consider citing our paper.

```bibtex
@article{Dong2025NeurCross,
author={Dong, Qiujie and Wen, Huibiao and Xu, Rui and Chen, Shuangmin and Zhou, Jiaran and Xin, Shiqing and Tu, Changhe and Komura, Taku and Wang, Wenping},
title={NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation},
journal={ACM Trans. Graph.},
publisher={Association for Computing Machinery},
address={New York, NY, USA},
year={2025},
volume={44},
number={4},
url={https://doi.org/10.1145/3731159},
doi={10.1145/3731159}
}
```



## Acknowledgments

Our code is inspired by [NeurCADRecon](https://github.com/QiujieDong/NeurCADRecon)
and [SIREN](https://github.com/vsitzmann/siren).
We would like to thank the authors
of [pyquadwild](https://github.com/dickoah/pyquadwild).
