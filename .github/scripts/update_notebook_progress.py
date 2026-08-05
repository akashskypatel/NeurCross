from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("NeurCross_Training.ipynb")

DASHBOARD_CODE = r'''
_PROGRESS_LINE_RE = re.compile(
    r"\[(?P<bar>[█░#-]+)\]\s*(?P<percent>\d+)%\s*"
    r"\(\s*(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*\)\s*\|\s*"
    r"Epoch:\s*(?P<epoch>\d+)\s+Loss:\s*(?P<loss>[-+0-9.eE]+)"
)
_IDENTITY_LINE_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*"
    r"\[(?P<identity>pid=[^\]]+)\]"
)
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
NOTEBOOK_PROGRESS_MIN_REFRESH_SECONDS = 0.20
NOTEBOOK_OUTPUT_LOCK = threading.RLock()


def notebook_print(*args, **kwargs) -> None:
    with NOTEBOOK_OUTPUT_LOCK:
        print(*args, **kwargs)


def _device_sort_key(device: str):
    text = str(device).strip().lower()
    if text == "cpu":
        return (1, 0, text)
    if text == "cuda":
        return (0, 0, text)
    if text.startswith("cuda:"):
        try:
            return (0, int(text.split(":", 1)[1]), text)
        except ValueError:
            pass
    return (2, 0, text)


class NotebookTrainingDashboard:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._display_id = f"neurcross-training-{uuid.uuid4().hex}"
        self._display_handle = None
        self._displayed = False
        self._last_render_monotonic = 0.0
        self._source_id = None
        self._states: dict[str, dict] = {}

    def start(self, devices: list[str]) -> None:
        with self._lock:
            for device in devices:
                self._state_for(device)
            self._render_locked(force=True)

    def set_source(self, source_id: str) -> None:
        with self._lock:
            self._source_id = source_id
            self._render_locked(force=True)

    def begin(self, device: str, sample_id: str, mesh_name: str | None = None) -> None:
        with self._lock:
            state = self._state_for(device)
            state.update(
                {
                    "sample_id": sample_id,
                    "mesh_name": mesh_name or sample_id,
                    "percent": 0,
                    "step": 0,
                    "total": 0,
                    "epoch": 0,
                    "loss": None,
                    "identity": None,
                    "status": "starting",
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                }
            )
            self._render_locked(force=True)

    def update_progress(
        self,
        *,
        device: str,
        sample_id: str,
        percent: int,
        step: int,
        total: int,
        epoch: int,
        loss: float,
        identity: str | None,
    ) -> None:
        with self._lock:
            state = self._state_for(device)
            state.update(
                {
                    "sample_id": sample_id,
                    "percent": max(0, min(100, int(percent))),
                    "step": max(0, int(step)),
                    "total": max(0, int(total)),
                    "epoch": max(0, int(epoch)),
                    "loss": float(loss),
                    "identity": identity,
                    "status": "training",
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                }
            )
            self._render_locked()

    def set_status(self, device: str, sample_id: str | None, status: str) -> None:
        with self._lock:
            state = self._state_for(device)
            if sample_id:
                state["sample_id"] = sample_id
            state["status"] = status
            state["updated_at"] = datetime.now().strftime("%H:%M:%S")
            self._render_locked(force=True)

    def finish(
        self,
        device: str,
        sample_id: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        final_status = status if not error else f"{status}: {error}"
        self.set_status(device, sample_id, final_status)

    def _state_for(self, device: str) -> dict:
        device = str(device)
        return self._states.setdefault(
            device,
            {
                "device": device,
                "sample_id": None,
                "mesh_name": None,
                "percent": 0,
                "step": 0,
                "total": 0,
                "epoch": 0,
                "loss": None,
                "identity": None,
                "status": "idle",
                "updated_at": "",
            },
        )

    def _html_locked(self) -> str:
        source_text = escape(str(self._source_id or "waiting for source"))
        rows = []
        for device in sorted(self._states, key=_device_sort_key):
            state = self._states[device]
            percent = int(state.get("percent", 0))
            bar_width = 42
            filled = max(0, min(bar_width, round(percent * bar_width / 100.0)))
            bar = "█" * filled + "░" * (bar_width - filled)
            step = int(state.get("step", 0))
            total = int(state.get("total", 0))
            step_text = f"{step}/{total}" if total > 0 else "-"
            loss = state.get("loss")
            loss_text = "-" if loss is None else f"{float(loss):.5f}"
            sample_text = escape(str(state.get("sample_id") or "-"))
            status_text = escape(str(state.get("status") or ""))
            identity_text = escape(str(state.get("identity") or ""))
            rows.append(
                "<tr>"
                f"<td><strong>{escape(device)}</strong></td>"
                f"<td>{sample_text}</td>"
                f"<td><code>[{bar}] {percent:3d}%</code></td>"
                f"<td>{escape(step_text)}</td>"
                f"<td>{int(state.get('epoch', 0))}</td>"
                f"<td>{loss_text}</td>"
                f"<td>{status_text}</td>"
                f"<td><small>{identity_text}</small></td>"
                f"<td>{escape(str(state.get('updated_at') or ''))}</td>"
                "</tr>"
            )
        rows_html = "".join(rows) or "<tr><td colspan='9'>No training devices detected.</td></tr>"
        return (
            "<div style='font-family:system-ui,sans-serif'>"
            "<div style='margin:0 0 8px 0'>"
            "<strong>NeurCross multi-device training</strong>"
            f"<span style='margin-left:12px'>Source: <code>{source_text}</code></span>"
            "</div>"
            "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            "<thead><tr>"
            "<th style='text-align:left'>Device</th>"
            "<th style='text-align:left'>Sample</th>"
            "<th style='text-align:left'>Progress</th>"
            "<th style='text-align:left'>Step</th>"
            "<th style='text-align:left'>Epoch</th>"
            "<th style='text-align:left'>Loss</th>"
            "<th style='text-align:left'>Status</th>"
            "<th style='text-align:left'>Process</th>"
            "<th style='text-align:left'>Updated</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "</div>"
        )

    def _render_locked(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render_monotonic < NOTEBOOK_PROGRESS_MIN_REFRESH_SECONDS:
            return
        payload = HTML(self._html_locked())
        try:
            if not self._displayed:
                self._display_handle = display(payload, display_id=self._display_id)
                self._displayed = True
            elif self._display_handle is not None:
                self._display_handle.update(payload)
            else:
                update_display(payload, display_id=self._display_id)
        except Exception as exc:
            notebook_print(f"Notebook progress display update failed: {exc}", flush=True)
        self._last_render_monotonic = now


NOTEBOOK_TRAINING_DASHBOARD = NotebookTrainingDashboard()
'''.lstrip()

NEW_RUN_SUBPROCESS = r'''
def run_subprocess_streaming(
    cmd: list[str],
    *,
    heartbeat=None,
    progress_device: str | None = None,
    progress_sample_id: str | None = None,
) -> None:
    """Run neurcross and aggregate each child process into the notebook dashboard."""
    resolved_env = os.environ.copy()
    resolved_env["PYTHONUNBUFFERED"] = "1"
    resolved_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    resolved_env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    device_label = str(progress_device or "worker")
    sample_label = str(progress_sample_id or "")
    notebook_print(f"[{device_label}] Running command:", flush=True)
    notebook_print(f"[{device_label}] {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=resolved_env,
    )
    assert process.stdout is not None
    next_heartbeat = datetime.now(timezone.utc) + timedelta(seconds=CLAIM_HEARTBEAT_SECONDS)
    pending_identity = None

    def consume_fragment(fragment: str) -> None:
        nonlocal pending_identity
        fragment = _ANSI_ESCAPE_RE.sub("", fragment).strip()
        if not fragment:
            return
        identity_match = _IDENTITY_LINE_RE.search(fragment)
        progress_match = _PROGRESS_LINE_RE.search(fragment)
        if identity_match:
            pending_identity = identity_match.group("identity")
        if progress_match and progress_device is not None:
            NOTEBOOK_TRAINING_DASHBOARD.update_progress(
                device=progress_device,
                sample_id=sample_label,
                percent=int(progress_match.group("percent")),
                step=int(progress_match.group("step")),
                total=int(progress_match.group("total")),
                epoch=int(progress_match.group("epoch")),
                loss=float(progress_match.group("loss")),
                identity=pending_identity,
            )
            return
        if identity_match and identity_match.group(0) == fragment:
            return
        if progress_device is not None:
            NOTEBOOK_TRAINING_DASHBOARD.set_status(progress_device, sample_label, fragment)
        notebook_print(f"[{device_label}] {fragment}", flush=True)

    fragment_chars: list[str] = []
    while True:
        char = process.stdout.read(1)
        if char == "":
            if fragment_chars:
                consume_fragment("".join(fragment_chars))
            break
        if char in {"\r", "\n"}:
            if fragment_chars:
                consume_fragment("".join(fragment_chars))
                fragment_chars.clear()
        else:
            fragment_chars.append(char)
        if heartbeat is not None and datetime.now(timezone.utc) >= next_heartbeat:
            try:
                heartbeat()
            except Exception as exc:
                notebook_print(f"[{device_label}] Lease heartbeat failed: {exc}", flush=True)
            next_heartbeat = datetime.now(timezone.utc) + timedelta(seconds=CLAIM_HEARTBEAT_SECONDS)

    process.stdout.close()
    return_code = process.wait()
    if heartbeat is not None:
        try:
            heartbeat()
        except Exception as exc:
            notebook_print(f"[{device_label}] Final lease heartbeat failed: {exc}", flush=True)
    if return_code != 0:
        if progress_device is not None:
            NOTEBOOK_TRAINING_DASHBOARD.finish(
                progress_device,
                sample_label,
                "training failed",
                f"exit code {return_code}",
            )
        raise subprocess.CalledProcessError(return_code, cmd)
    if progress_device is not None:
        NOTEBOOK_TRAINING_DASHBOARD.set_status(
            progress_device,
            sample_label,
            "training complete; packaging artifacts",
        )
'''.lstrip()


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source_text(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str, *, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {count}")
    return text.replace(old, new, 1)


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

helper_cell = next(cell for cell in notebook["cells"] if "def run_subprocess_streaming" in source_text(cell))
helper_source = source_text(helper_cell)
helper_source = replace_once(
    helper_source,
    "from IPython.display import clear_output\n",
    "from html import escape\nfrom IPython.display import HTML, display, update_display\n",
    description="IPython display import",
)
helper_source = replace_once(
    helper_source,
    "NEURCROSS_MODULE = \"neurcross\"\n",
    DASHBOARD_CODE + "\nNEURCROSS_MODULE = \"neurcross\"\n",
    description="dashboard insertion point",
)
run_start = helper_source.index("def run_subprocess_streaming(")
run_end = helper_source.index("def _rewrite_manifest_paths", run_start)
helper_source = helper_source[:run_start] + NEW_RUN_SUBPROCESS + "\n" + helper_source[run_end:]
set_source_text(helper_cell, helper_source)

process_cell = next(cell for cell in notebook["cells"] if "def process_mesh_entry" in source_text(cell))
process_source = source_text(process_cell)
process_source = replace_once(
    process_source,
    "    resume_upload_result = {\"uploaded\": False, \"path_in_repo\": None}\n",
    "    resume_upload_result = {\"uploaded\": False, \"path_in_repo\": None}\n"
    "    NOTEBOOK_TRAINING_DASHBOARD.begin(\n"
    "        training_device,\n"
    "        sample_id,\n"
    "        mesh_entry.get(\"mesh_relative_path\"),\n"
    "    )\n",
    description="per-device dashboard begin",
)
process_source = replace_once(
    process_source,
    "        run_subprocess_streaming(command, heartbeat=heartbeat_claim)\n",
    "        run_subprocess_streaming(\n"
    "            command,\n"
    "            heartbeat=heartbeat_claim,\n"
    "            progress_device=training_device,\n"
    "            progress_sample_id=sample_id,\n"
    "        )\n",
    description="dashboard-aware subprocess call",
)
process_source = replace_once(
    process_source,
    "    return final_status\n",
    "    NOTEBOOK_TRAINING_DASHBOARD.finish(\n"
    "        training_device,\n"
    "        sample_id,\n"
    "        final_status,\n"
    "        error_message,\n"
    "    )\n"
    "    return final_status\n",
    description="per-device dashboard final status",
)
process_source = replace_once(
    process_source,
    "    training_devices = list_training_devices()\n"
    "    disabled_devices = set()\n"
    "    print(f\"Training devices for this execution: {training_devices}\")\n",
    "    training_devices = list_training_devices()\n"
    "    disabled_devices = set()\n"
    "    NOTEBOOK_TRAINING_DASHBOARD.start(training_devices)\n"
    "    notebook_print(f\"Training devices for this execution: {training_devices}\")\n",
    description="dashboard startup",
)
process_source = replace_once(
    process_source,
    "    for source in DATA_SOURCES:\n"
    "        clear_output(wait=True)\n"
    "        if not unlimited_runs and processed_this_execution >= max_mesh_runs:\n"
    "            break\n"
    "        sid = source_id(source)\n",
    "    for source in DATA_SOURCES:\n"
    "        if not unlimited_runs and processed_this_execution >= max_mesh_runs:\n"
    "            break\n"
    "        sid = source_id(source)\n"
    "        NOTEBOOK_TRAINING_DASHBOARD.set_source(sid)\n",
    description="source dashboard update",
)
set_source_text(process_cell, process_source)

for cell in (helper_cell, process_cell):
    ast.parse(source_text(cell))

full_source = "\n".join(source_text(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code")
required_markers = (
    "class NotebookTrainingDashboard",
    "NOTEBOOK_TRAINING_DASHBOARD.start(training_devices)",
    "progress_device=training_device",
    "process.stdout.read(1)",
)
for marker in required_markers:
    if marker not in full_source:
        raise RuntimeError(f"Notebook update is missing required marker: {marker}")
if "clear_output" in full_source:
    raise RuntimeError("Notebook still uses clear_output, which invalidates display-id progress updates")

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("Updated NeurCross_Training.ipynb with a multi-device in-place dashboard.")
