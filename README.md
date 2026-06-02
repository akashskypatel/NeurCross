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

Estimated runtime from the paper: for a triangular mesh with 50,000 faces, each optimization iteration takes about `68.34 ms`, and the default research setting uses `10,000` iterations. That corresponds to roughly `683.4` seconds, or about `11.4` minutes, for one full run under that configuration. Actual runtime in this repository will vary with GPU, PyTorch/CUDA version, mesh complexity, and your chosen `--n_samples` setting.

By default, training now also runs quad extraction at the end of optimization and writes an OBJ mesh beside the input mesh. For the sample input, the extracted mesh is written to:

```text
data/doubleTorus/input/doubleTorus/doubleTorus_quad.obj
```

The extraction step uses the predicted cross field, a libigl-based seam/UV solve, and `pyqex` quad extraction.

## `quad_mesh_args.py` Reference

The training entry point accepts the following arguments.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--logdir` | `./output/` | Output directory used for logs and training artifacts. The script creates a subdirectory named after the input mesh file. |
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
| `--no_extract_quad_mesh` | disabled | Disables the automatic quad mesh extraction step that runs after training. |
| `--quad_mesh_output` | `None` | Optional explicit OBJ output path for the extracted quad mesh. If omitted, the mesh is written beside the input mesh under a subdirectory named after the input file. |
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

Notes:

- Boolean flags defined with `action='store_true'` such as `--udf`, `--output_any`, `--morse_near`, and `--weight_for_morse` are disabled by default and become enabled when the flag is present.
- `--use_morse_nonmnfld_grad` and `--use_vertices` use `type=bool`, so if you pass them explicitly on the command line, use forms such as `--use_vertices True` or `--use_vertices False`.
- Some arguments are preserved from the original research code even when the current training path uses them lightly or not at all. The table above reflects the behavior of the current repository state.


## Extraction

The extraction stage in this repository uses [libigl](https://libigl.github.io/)
for cross-field combing, seam cutting, and UV construction, and [libQEx](https://github.com/hcebke/libQEx)
through the Python `pyqex` binding for quad extraction.


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
of [libigl](https://libigl.github.io/)
and [libQEx](https://github.com/hcebke/libQEx).

