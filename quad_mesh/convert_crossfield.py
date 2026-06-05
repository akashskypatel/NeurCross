from __future__ import annotations

import argparse
from pathlib import Path


def infer_rosy_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".rosy")


def infer_rawfield_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".rawfield")


def _load_crossfield_rows(input_path: Path) -> list[list[float]]:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input cross-field file was not found: {input_path}")

    rows: list[list[float]] = []
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
                rows.append([float(value) for value in parts])
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} in {input_path} contains non-numeric cross-field data."
                ) from exc
    return rows


def convert_crossfield_to_rosy(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert a NeurCross cross-field text file to QuadWild's .rosy format.

    The input file is expected to contain one face per line, with at least
    three floating-point values for the first cross-field direction.
    """
    input_path = Path(input_path)
    output_path = infer_rosy_output_path(input_path) if output_path is None else Path(output_path)

    rows = _load_crossfield_rows(input_path)
    vectors = [(row[0], row[1], row[2]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{len(vectors)}\n")
        outfile.write("4\n")
        for x, y, z in vectors:
            outfile.write(f"{x} {y} {z}\n")

    return output_path


def convert_crossfield_to_rawfield(
    input_path: Path,
    output_path: Path | None = None,
    *,
    degree: int = 4,
) -> Path:
    """
    Convert a NeurCross cross-field text file to Directional's .rawfield format.

    The input must contain at least 6 values per line:
    the primary branch (alpha xyz) followed by the secondary branch (beta xyz).
    """
    input_path = Path(input_path)
    output_path = infer_rawfield_output_path(input_path) if output_path is None else Path(output_path)

    rows = _load_crossfield_rows(input_path)
    raw_rows: list[tuple[float, ...]] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) < 6:
            raise ValueError(
                f"Line {line_number} in {input_path} has {len(row)} values; expected at least 6 for rawfield export."
            )
        ax, ay, az, bx, by, bz = row[:6]
        raw_rows.append((ax, ay, az, bx, by, bz, -ax, -ay, -az, -bx, -by, -bz))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{int(degree)} {len(raw_rows)}\n")
        for raw_row in raw_rows:
            outfile.write(" ".join(str(value) for value in raw_row))
            outfile.write("\n")

    return output_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a NeurCross cross-field text file into .rosy or Directional .rawfield format."
    )
    parser.add_argument("input_path", type=Path, help="Path to the NeurCross cross-field .txt file.")
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Optional output path. Defaults to the input path with a .rosy or .rawfield extension.",
    )
    parser.add_argument(
        "--format",
        choices=("rosy", "rawfield"),
        default="rosy",
        help="Output format to write.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.format == "rawfield":
        output_path = convert_crossfield_to_rawfield(args.input_path, args.output_path)
    else:
        output_path = convert_crossfield_to_rosy(args.input_path, args.output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
