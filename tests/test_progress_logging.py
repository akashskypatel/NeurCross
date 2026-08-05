import io
import sys

import utils.utils as utils


def test_training_metrics_render_as_static_progress(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)

    log_file = io.StringIO()
    log_file._neurcross_log_identity = "pid=3082 device=cuda:0 sample=icecream"

    utils.log_string(
        "Training schedule: requested_n_samples=1000 derived_steps_per_epoch=1000 "
        "requested_num_epochs=10 derived_num_epochs=10 requested_total_steps=10000 "
        "n_points_per_batch=100 batch_size=1 total_steps=10000",
        log_file,
    )
    utils.log_string("Weights: [7000.0, 600.0], lr=5.000e-05", log_file)
    utils.log_string("Timing: batch=0.355s total_elapsed=36m 57.20s", log_file)
    utils.log_string(
        "Epoch: 0 [   9/1000 (1%)] Loss: 82.01839 = L_Mnfld: 8.73765",
        log_file,
    )
    utils.log_string(
        "Epoch: 0 [   9/1000 (1%)] Unweighted L_s : L_Mnfld: 0.00125",
        log_file,
    )
    utils.log_string("", log_file)

    console_output = console.getvalue()
    file_output = log_file.getvalue()

    assert "Weights:" not in console_output
    assert "Timing:" not in console_output
    assert "Unweighted L_s" not in console_output
    assert "[█" in console_output
    assert "1% ( 10/10000) | Epoch: 0 Loss: 82.01839" in console_output
    assert "\r" in console_output

    assert "Weights:" in file_output
    assert "Timing:" in file_output
    assert "Unweighted L_s" in file_output


def test_normal_log_message_preserves_active_progress(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)

    log_file = io.StringIO()
    log_file._neurcross_log_identity = "pid=1 device=cpu sample=test"

    utils.log_string(
        "Training schedule: requested_n_samples=10 derived_steps_per_epoch=10 "
        "requested_num_epochs=1 derived_num_epochs=1 requested_total_steps=10 "
        "n_points_per_batch=10 batch_size=1 total_steps=10",
        log_file,
    )
    utils.log_string("Epoch: 0 [   4/10 (40%)] Loss: 5.00000", log_file)
    utils.log_string("Validation: global_step=5 score=0.250000", log_file)
    utils.log_string("Training finished in 1.000s", log_file)

    console_output = console.getvalue()

    assert "Validation: global_step=5 score=0.250000" in console_output
    assert console_output.count("Epoch: 0 Loss: 5.00000") == 2
    assert "Training finished in 1.000s" in console_output
    assert console_output.endswith("\n")
