def train_crossfield(*, argv=None, args=None, allow_multiprocessing_workers=False, **overrides):
    from .train_quad_mesh import train_crossfield as _train_crossfield

    return _train_crossfield(
        argv=argv,
        args=args,
        allow_multiprocessing_workers=allow_multiprocessing_workers,
        **overrides,
    )


def convert_crossfield_to_rosy(input_path, output_path=None):
    from .convert_crossfield import convert_crossfield_to_rosy as _convert_crossfield_to_rosy

    return _convert_crossfield_to_rosy(input_path, output_path)


def convert_crossfield_to_rawfield(input_path, output_path=None, *, degree=4):
    from .convert_crossfield import convert_crossfield_to_rawfield as _convert_crossfield_to_rawfield

    return _convert_crossfield_to_rawfield(input_path, output_path, degree=degree)


def convert_rawfield_to_rosy(input_path, output_path=None):
    from .convert_crossfield import convert_rawfield_to_rosy as _convert_rawfield_to_rosy

    return _convert_rawfield_to_rosy(input_path, output_path)


def __getattr__(name):
    if name == "TrainingResult":
        from .train_quad_mesh import TrainingResult

        return TrainingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrainingResult",
    "convert_crossfield_to_rawfield",
    "convert_crossfield_to_rosy",
    "convert_rawfield_to_rosy",
    "train_crossfield",
]
