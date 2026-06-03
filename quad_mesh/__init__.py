from pathlib import Path


def convert_crossfield(input_path: Path, output_path: Path | None = None):
    from .crossfield_to_rosy import convert_crossfield as _convert_crossfield

    return _convert_crossfield(input_path, output_path)


__all__ = ["convert_crossfield"]
