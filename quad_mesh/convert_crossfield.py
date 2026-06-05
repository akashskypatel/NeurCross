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


def _load_rawfield_rows(input_path: Path) -> tuple[int, list[list[float]]]:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input rawfield file was not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as infile:
        lines = [line.strip() for line in infile if line.strip()]

    if not lines:
        raise ValueError(f"Input rawfield file is empty: {input_path}")

    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(
            f"Rawfield header in {input_path} must contain exactly 2 values: '<degree> <count>'."
        )

    try:
        degree = int(header[0])
        count = int(header[1])
    except ValueError as exc:
        raise ValueError(
            f"Rawfield header in {input_path} contains non-integer values."
        ) from exc

    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        parts = raw_line.split()
        if len(parts) < 3:
            raise ValueError(
                f"Line {line_number} in {input_path} has {len(parts)} values; expected at least 3."
            )
        try:
            rows.append([float(value) for value in parts])
        except ValueError as exc:
            raise ValueError(
                f"Line {line_number} in {input_path} contains non-numeric rawfield data."
            ) from exc

    if len(rows) != count:
        raise ValueError(
            f"Rawfield header in {input_path} declares {count} rows but file contains {len(rows)} rows."
        )

    return degree, rows


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


def convert_rawfield_to_rosy(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert a Directional .rawfield file to QuadWild's .rosy format.

    The input is expected to follow Directional's headered rawfield format:
    first line '<degree> <num_faces>', followed by one row per face.
    The first 3 values of each row are used as the primary RoSy direction.
    """
    input_path = Path(input_path)
    output_path = infer_rosy_output_path(input_path) if output_path is None else Path(output_path)

    degree, rows = _load_rawfield_rows(input_path)
    if degree <= 0:
        raise ValueError(f"Rawfield degree must be positive in {input_path}.")

    vectors = [(row[0], row[1], row[2]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{len(vectors)}\n")
        outfile.write(f"{degree}\n")
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
        description="Convert NeurCross cross-field text files and Directional rawfield files into .rosy or .rawfield."
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
    parser.add_argument(
        "--input-format",
        choices=("auto", "crossfield", "rawfield"),
        default="auto",
        help="Input format. 'auto' infers from the input extension.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_format = args.input_format
    if input_format == "auto":
        input_format = "rawfield" if args.input_path.suffix.lower() == ".rawfield" else "crossfield"

    if args.format == "rawfield":
        if input_format != "crossfield":
            raise ValueError("Rawfield output requires a crossfield text input.")
        output_path = convert_crossfield_to_rawfield(args.input_path, args.output_path)
    else:
        if input_format == "rawfield":
            output_path = convert_rawfield_to_rosy(args.input_path, args.output_path)
        else:
            output_path = convert_crossfield_to_rosy(args.input_path, args.output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
