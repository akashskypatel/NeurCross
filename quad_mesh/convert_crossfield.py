from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def infer_rosy_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".rosy")


def infer_rawfield_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".rawfield")


def infer_crossfield_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".vec")


def infer_output_path(input_path: Path, output_format: str) -> Path:
    if output_format == "rosy":
        return infer_rosy_output_path(input_path)
    if output_format == "rawfield":
        return infer_rawfield_output_path(input_path)
    if output_format == "crossfield":
        return infer_crossfield_output_path(input_path)
    raise ValueError(f"Unsupported output format: {output_format}")


def _warn_lossy(message: str) -> None:
    print(f"WARNING: lossy conversion: {message}", file=sys.stderr)


def _strip_comment(raw_line: str) -> str:
    return raw_line.split("#", 1)[0].strip()


def _split_numeric_line(raw_line: str) -> list[str]:
    return _strip_comment(raw_line).replace(",", " ").split()


def _parse_float_row(parts: list[str], *, line_number: int, input_path: Path) -> list[float]:
    try:
        return [float(value) for value in parts]
    except ValueError as exc:
        raise ValueError(
            f"Line {line_number} in {input_path} contains non-numeric field data."
        ) from exc


def _load_crossfield_rows(input_path: Path) -> list[list[float]]:
    """
    Load NeurCross-style crossfield rows.

    Expected canonical layout:
        ax ay az bx by bz

    A CSV header such as:
        face_id,ax,ay,az,bx,by,bz

    is also tolerated. If a numeric face_id column is present with 7+ values,
    it is removed automatically.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input cross-field file was not found: {input_path}")

    rows: list[list[float]] = []
    skipped_header = False
    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            parts = _split_numeric_line(raw_line)
            if not parts:
                continue

            try:
                row = [float(value) for value in parts]
            except ValueError:
                if not rows and not skipped_header:
                    skipped_header = True
                    continue
                raise ValueError(
                    f"Line {line_number} in {input_path} contains non-numeric cross-field data."
                )

            # Optional face_id column: face_id ax ay az bx by bz [...]
            if len(row) >= 7 and abs(row[0] - round(row[0])) < 1e-9:
                # Only strip if it looks like a monotonically aligned row id.
                expected_id = len(rows)
                if int(round(row[0])) == expected_id:
                    row = row[1:]

            if len(row) < 3:
                raise ValueError(
                    f"Line {line_number} in {input_path} has {len(row)} values; expected at least 3."
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"Input cross-field file contains no numeric rows: {input_path}")
    return rows


def _load_rawfield_rows(input_path: Path) -> tuple[int, list[list[float]]]:
    """
    Load this script's 3D rawfield interchange format.

    Header:
        <degree> <num_faces>

    Rows:
        degree * 3 values per face:
        x0 y0 z0 x1 y1 z1 ...
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input rawfield file was not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as infile:
        lines = [_strip_comment(line) for line in infile]
    lines = [line for line in lines if line]

    if not lines:
        raise ValueError(f"Input rawfield file is empty: {input_path}")

    header = lines[0].replace(",", " ").split()
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

    if degree <= 0:
        raise ValueError(f"Rawfield degree must be positive in {input_path}.")
    if count < 0:
        raise ValueError(f"Rawfield count must be non-negative in {input_path}.")

    expected_values = degree * 3
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        parts = raw_line.replace(",", " ").split()
        if len(parts) != expected_values:
            raise ValueError(
                f"Line {line_number} in {input_path} has {len(parts)} values; "
                f"expected exactly {expected_values} for degree {degree}."
            )
        rows.append(_parse_float_row(parts, line_number=line_number, input_path=input_path))

    if len(rows) != count:
        raise ValueError(
            f"Rawfield header in {input_path} declares {count} rows but file contains {len(rows)} rows."
        )

    return degree, rows


def _load_rosy_rows(input_path: Path) -> tuple[int, list[tuple[float, float, float]]]:
    """
    Load QuadWild .rosy format.

    Header:
        <num_faces>
        <degree>

    Rows:
        one xyz representative direction per face.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input rosy file was not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as infile:
        lines = [_strip_comment(line) for line in infile]
    lines = [line for line in lines if line]

    if len(lines) < 2:
        raise ValueError(f"Invalid .rosy file: {input_path} is missing its two-line header.")

    try:
        count = int(lines[0].strip())
        degree = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid .rosy header in {input_path}.") from exc

    rows: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(lines[2:], start=3):
        parts = raw_line.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(
                f"Line {line_number} in {input_path} has {len(parts)} values; expected exactly 3."
            )
        x, y, z = _parse_float_row(parts, line_number=line_number, input_path=input_path)
        rows.append((x, y, z))

    if len(rows) != count:
        raise ValueError(
            f"Rosy header in {input_path} declares {count} rows but file contains {len(rows)} rows."
        )
    if degree <= 0:
        raise ValueError(f"Rosy degree must be positive in {input_path}.")

    return degree, rows


def _normalize(v: tuple[float, float, float], eps: float = 1e-12) -> tuple[float, float, float]:
    x, y, z = v
    n = (x * x + y * y + z * z) ** 0.5
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return (x / n, y / n, z / n)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _project_to_tangent(
    v: tuple[float, float, float],
    n: tuple[float, float, float],
) -> tuple[float, float, float]:
    d = _dot(v, n)
    return _normalize((v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]))


def _load_mesh_face_normals(mesh_path: Path, *, expected_faces: int) -> list[tuple[float, float, float]]:
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Converting from .rosy to crossfield/rawfield requires --mesh and the 'trimesh' package. "
            "Install it with: pip install trimesh"
        ) from exc

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not hasattr(mesh, "faces") or not hasattr(mesh, "face_normals"):
        raise ValueError(f"Mesh did not load as a single triangle mesh: {mesh_path}")

    if len(mesh.faces) != expected_faces:
        raise ValueError(
            f"Face count mismatch: mesh has {len(mesh.faces)} faces but field has {expected_faces} rows."
        )

    return [_normalize(tuple(map(float, normal))) for normal in mesh.face_normals]


def _write_crossfield_rows(output_path: Path, rows: list[tuple[float, ...]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        for row in rows:
            outfile.write(" ".join(f"{value:.17g}" for value in row))
            outfile.write("\n")
    return output_path


def _write_rawfield_rows(output_path: Path, degree: int, rows: list[tuple[float, ...]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{degree} {len(rows)}\n")
        for row in rows:
            outfile.write(" ".join(f"{value:.17g}" for value in row))
            outfile.write("\n")
    return output_path


def _write_rosy_rows(
    output_path: Path,
    degree: int,
    vectors: list[tuple[float, float, float]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{len(vectors)}\n")
        outfile.write(f"{degree}\n")
        for x, y, z in vectors:
            outfile.write(f"{x:.17g} {y:.17g} {z:.17g}\n")
    return output_path


def _crossfield_rows_to_raw_rows(
    rows: list[list[float]],
    *,
    degree: int = 4,
    input_path: Path | None = None,
) -> list[tuple[float, ...]]:
    if degree not in (2, 4):
        raise ValueError("Only degree 2 or 4 rawfield export is supported from crossfield input.")

    raw_rows: list[tuple[float, ...]] = []
    ignored_extra_columns = False
    for line_number, row in enumerate(rows, start=1):
        if len(row) < 6:
            source = f" in {input_path}" if input_path is not None else ""
            raise ValueError(
                f"Line {line_number}{source} has {len(row)} values; expected at least 6 for rawfield/crossfield export."
            )
        if len(row) > 6:
            ignored_extra_columns = True
        ax, ay, az, bx, by, bz = row[:6]
        if degree == 2:
            raw_rows.append((ax, ay, az, bx, by, bz))
        else:
            raw_rows.append((ax, ay, az, bx, by, bz, -ax, -ay, -az, -bx, -by, -bz))

    if ignored_extra_columns:
        _warn_lossy("ignored extra columns after ax ay az bx by bz in crossfield input")
    return raw_rows


def convert_crossfield_to_rosy(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert NeurCross crossfield text to QuadWild .rosy.

    This is lossy when the input has beta columns because .rosy stores only one
    representative vector per face.
    """
    input_path = Path(input_path)
    output_path = infer_rosy_output_path(input_path) if output_path is None else Path(output_path)

    rows = _load_crossfield_rows(input_path)
    if any(len(row) >= 6 for row in rows):
        _warn_lossy("crossfield -> rosy drops beta and keeps only alpha")
    vectors = [_normalize((row[0], row[1], row[2])) for row in rows]
    return _write_rosy_rows(output_path, 4, vectors)


def convert_crossfield_to_rawfield(
    input_path: Path,
    output_path: Path | None = None,
    *,
    degree: int = 4,
) -> Path:
    """
    Convert NeurCross crossfield text to this script's 3D .rawfield interchange format.

    Canonical input row:
        ax ay az bx by bz

    degree=4 output row:
        alpha beta -alpha -beta
    """
    input_path = Path(input_path)
    output_path = infer_rawfield_output_path(input_path) if output_path is None else Path(output_path)

    rows = _load_crossfield_rows(input_path)
    raw_rows = _crossfield_rows_to_raw_rows(rows, degree=degree, input_path=input_path)
    return _write_rawfield_rows(output_path, degree, raw_rows)


def convert_crossfield_to_crossfield(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = Path(input_path)
    output_path = infer_crossfield_output_path(input_path) if output_path is None else Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths are identical for crossfield -> crossfield conversion.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    return output_path


def convert_rawfield_to_rosy(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert 3D .rawfield to QuadWild .rosy.

    Lossy for degree > 1 because .rosy stores only the first branch per face.
    """
    input_path = Path(input_path)
    output_path = infer_rosy_output_path(input_path) if output_path is None else Path(output_path)

    degree, rows = _load_rawfield_rows(input_path)
    if degree > 1:
        _warn_lossy("rawfield -> rosy drops all branches except the first vector")
    vectors = [_normalize((row[0], row[1], row[2])) for row in rows]
    return _write_rosy_rows(output_path, degree, vectors)


def convert_rawfield_to_crossfield(
    input_path: Path,
    output_path: Path | None = None,
    *,
    mesh_path: Path | None = None,
) -> Path:
    """
    Convert 3D .rawfield to NeurCross-style crossfield text.

    If degree >= 2, the first two branches become alpha and beta.
    If degree == 1, --mesh is required to reconstruct beta from face normals.
    """
    input_path = Path(input_path)
    output_path = infer_crossfield_output_path(input_path) if output_path is None else Path(output_path)

    degree, rows = _load_rawfield_rows(input_path)
    cross_rows: list[tuple[float, ...]] = []

    if degree >= 2:
        if degree > 2:
            _warn_lossy("rawfield -> crossfield keeps only the first two branches")
        for row in rows:
            alpha = _normalize((row[0], row[1], row[2]))
            beta = _normalize((row[3], row[4], row[5]))
            cross_rows.append((*alpha, *beta))
    else:
        if mesh_path is None:
            raise ValueError("rawfield degree 1 -> crossfield requires --mesh to reconstruct beta.")
        _warn_lossy("rawfield degree 1 -> crossfield reconstructs beta from mesh normals")
        normals = _load_mesh_face_normals(Path(mesh_path), expected_faces=len(rows))
        for row, normal in zip(rows, normals):
            alpha = _project_to_tangent(_normalize((row[0], row[1], row[2])), normal)
            beta = _normalize(_cross(normal, alpha))
            cross_rows.append((*alpha, *beta))

    return _write_crossfield_rows(output_path, cross_rows)


def convert_rawfield_to_rawfield(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = Path(input_path)
    output_path = infer_rawfield_output_path(input_path) if output_path is None else Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths are identical for rawfield -> rawfield conversion.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    return output_path


def _rosy_to_cross_rows(
    input_path: Path,
    *,
    mesh_path: Path,
) -> tuple[int, list[tuple[float, ...]]]:
    degree, vectors = _load_rosy_rows(input_path)
    normals = _load_mesh_face_normals(Path(mesh_path), expected_faces=len(vectors))
    _warn_lossy(
        "rosy -> crossfield/rawfield reconstructs beta from mesh normals; original beta/sign/order are not recoverable"
    )

    cross_rows: list[tuple[float, ...]] = []
    for vector, normal in zip(vectors, normals):
        alpha = _project_to_tangent(_normalize(vector), normal)
        # beta = n x alpha, so alpha x beta = n for unit tangent alpha.
        beta = _normalize(_cross(normal, alpha))
        cross_rows.append((*alpha, *beta))
    return degree, cross_rows


def convert_rosy_to_crossfield(
    input_path: Path,
    output_path: Path | None = None,
    *,
    mesh_path: Path | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = infer_crossfield_output_path(input_path) if output_path is None else Path(output_path)
    if mesh_path is None:
        raise ValueError("rosy -> crossfield requires --mesh to reconstruct beta from face normals.")

    _degree, cross_rows = _rosy_to_cross_rows(input_path, mesh_path=Path(mesh_path))
    return _write_crossfield_rows(output_path, cross_rows)


def convert_rosy_to_rawfield(
    input_path: Path,
    output_path: Path | None = None,
    *,
    mesh_path: Path | None = None,
    degree: int = 4,
) -> Path:
    input_path = Path(input_path)
    output_path = infer_rawfield_output_path(input_path) if output_path is None else Path(output_path)
    if mesh_path is None:
        raise ValueError("rosy -> rawfield requires --mesh to reconstruct beta from face normals.")
    if degree not in (2, 4):
        raise ValueError("Only degree 2 or 4 rawfield export is supported from rosy input.")

    _rosy_degree, cross_rows = _rosy_to_cross_rows(input_path, mesh_path=Path(mesh_path))
    raw_rows = _crossfield_rows_to_raw_rows([list(row) for row in cross_rows], degree=degree)
    return _write_rawfield_rows(output_path, degree, raw_rows)


def convert_rosy_to_rosy(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = Path(input_path)
    output_path = infer_rosy_output_path(input_path) if output_path is None else Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths are identical for rosy -> rosy conversion.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert between NeurCross crossfield text, QuadWild .rosy, and a 3D rawfield "
            "interchange format. Lossy conversions print a warning to stderr."
        )
    )
    parser.add_argument("input_path", type=Path, help="Input field path.")
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Optional output path. Defaults to an extension inferred from --format.",
    )
    parser.add_argument(
        "--format",
        choices=("crossfield", "rosy", "rawfield"),
        default="rosy",
        help="Output format to write.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "crossfield", "rosy", "rawfield"),
        default="auto",
        help="Input format. 'auto' infers from the input extension.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help=(
            "Matching mesh path. Required for rosy -> crossfield/rawfield, and for rawfield degree 1 -> crossfield. "
            "The mesh face order must match the field row order."
        ),
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        choices=(2, 4),
        help="Rawfield degree to write for crossfield/rosy -> rawfield. Default: 4.",
    )
    return parser


def infer_input_format(input_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = input_path.suffix.lower()
    if suffix == ".rawfield":
        return "rawfield"
    if suffix == ".rosy":
        return "rosy"
    return "crossfield"


def convert(
    input_path: Path,
    output_path: Path | None,
    *,
    input_format: str,
    output_format: str,
    mesh_path: Path | None = None,
    degree: int = 4,
) -> Path:
    if output_path is None:
        output_path = infer_output_path(input_path, output_format)

    if input_format == "crossfield" and output_format == "crossfield":
        return convert_crossfield_to_crossfield(input_path, output_path)
    if input_format == "crossfield" and output_format == "rosy":
        return convert_crossfield_to_rosy(input_path, output_path)
    if input_format == "crossfield" and output_format == "rawfield":
        return convert_crossfield_to_rawfield(input_path, output_path, degree=degree)

    if input_format == "rawfield" and output_format == "crossfield":
        return convert_rawfield_to_crossfield(input_path, output_path, mesh_path=mesh_path)
    if input_format == "rawfield" and output_format == "rosy":
        return convert_rawfield_to_rosy(input_path, output_path)
    if input_format == "rawfield" and output_format == "rawfield":
        return convert_rawfield_to_rawfield(input_path, output_path)

    if input_format == "rosy" and output_format == "crossfield":
        return convert_rosy_to_crossfield(input_path, output_path, mesh_path=mesh_path)
    if input_format == "rosy" and output_format == "rawfield":
        return convert_rosy_to_rawfield(input_path, output_path, mesh_path=mesh_path, degree=degree)
    if input_format == "rosy" and output_format == "rosy":
        return convert_rosy_to_rosy(input_path, output_path)

    raise ValueError(f"Unsupported conversion: {input_format} -> {output_format}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_format = infer_input_format(args.input_path, args.input_format)

    output_path = convert(
        args.input_path,
        args.output_path,
        input_format=input_format,
        output_format=args.format,
        mesh_path=args.mesh,
        degree=args.degree,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
