from __future__ import annotations

import argparse
from pathlib import Path


def infer_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".rosy")


def convert_crossfield(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert a NeurCross cross-field text file to QuadWild's .rosy format.

    The input file is expected to contain one face per line, with at least
    three floating-point values for the first cross-field direction and
    optionally three more values for the orthogonal branch.
    """
    input_path = Path(input_path)
    output_path = infer_output_path(input_path) if output_path is None else Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input cross-field file was not found: {input_path}")

    vectors: list[tuple[float, float, float]] = []
    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Line {line_number} in {input_path} has {len(parts)} values; expected at least 3."
                )

            try:
                x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} in {input_path} contains non-numeric cross-field data."
                ) from exc

            vectors.append((x, y, z))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{len(vectors)}\n")
        outfile.write("4\n")
        for x, y, z in vectors:
            outfile.write(f"{x} {y} {z}\n")

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a NeurCross cross-field text file into QuadWild .rosy format."
    )
    parser.add_argument("input_path", type=Path, help="Path to the NeurCross cross-field .txt file.")
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Optional output .rosy path. Defaults to the input path with a .rosy extension.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = convert_crossfield(args.input_path, args.output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
