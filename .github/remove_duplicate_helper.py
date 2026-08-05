import ast
import json
from pathlib import Path

path = Path("NeurCross_Training.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
target = next(
    cell for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
    and "class NotebookTrainingDashboard" in "".join(cell.get("source", []))
)
source = "".join(target["source"])
duplicate = '''    return None


def notebook_print(*args, **kwargs) -> None:
    with NOTEBOOK_OUTPUT_LOCK:
        print(*args, **kwargs)


def _device_sort_key(device: str):
'''
replacement = '''    return None


def _device_sort_key(device: str):
'''
if source.count("def notebook_print(*args, **kwargs) -> None:") != 2:
    raise RuntimeError("expected exactly two notebook_print definitions")
if duplicate not in source:
    raise RuntimeError("duplicate helper block not found")
source = source.replace(duplicate, replacement, 1)
target["source"] = source.splitlines(keepends=True)
path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
reloaded = json.loads(path.read_text(encoding="utf-8"))
verified = "".join(target["source"])
ast.parse(verified)
assert verified.count("def notebook_print(*args, **kwargs) -> None:") == 1
assert len(reloaded["cells"]) == len(notebook["cells"])
print("Duplicate notebook helper removed and syntax verified.")
