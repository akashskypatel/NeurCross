__version__ = "0.1.0"

from quad_mesh.train_quad_mesh import TrainingResult


def train_crossfield(*, argv=None, args=None, **overrides):
    from quad_mesh.train_quad_mesh import train_crossfield as _train_crossfield

    return _train_crossfield(argv=argv, args=args, **overrides)


def convert_crossfield_to_rosy(input_path, output_path=None):
    from quad_mesh.convert_crossfield import convert_crossfield_to_rosy as _convert_crossfield_to_rosy

    return _convert_crossfield_to_rosy(input_path, output_path)


def convert_crossfield_to_rawfield(input_path, output_path=None, *, degree=4):
    from quad_mesh.convert_crossfield import (
        convert_crossfield_to_rawfield as _convert_crossfield_to_rawfield,
    )

    return _convert_crossfield_to_rawfield(input_path, output_path, degree=degree)

__all__ = [
    "TrainingResult",
    "__version__",
    "convert_crossfield_to_rawfield",
    "convert_crossfield_to_rosy",
    "train_crossfield",
]
