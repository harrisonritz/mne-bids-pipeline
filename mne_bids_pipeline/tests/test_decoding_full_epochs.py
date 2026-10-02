"""Tests for full-epochs decoding."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath
from scipy.io import loadmat

from mne_bids_pipeline.steps.sensor import _02_decoding_full_epochs as full_epochs

N_EPOCHS_PER_COND = 24
N_GROUPS = 6  # every group holds both conditions, as LOGO scoring requires
N_CH = 12
N_EFFECT_CH = 4  # the first channels carry the planted effect
SFREQ = 100.0
TMIN, TMAX = -0.1, 0.5  # -> 61 samples
EFFECT_TMIN = 0.3  # the planted effect is late, starting here
EFFECT_AMP = 1.0
UNIT = 1e-13  # magnetometer data in T, ~100 fT
GAIN = 1e3  # a change of the data units, to check how the weights scale


def _make_epochs(*, gain: float = 1.0, offset: float = 0.0) -> mne.EpochsArray:
    """Make two-condition magnetometer epochs with a planted late effect.

    The data are multiplied by ``gain`` and then shifted by the constant ``offset``.
    """
    rng = np.random.default_rng(0)
    ch_names = [f"MAG{ii:03d}" for ii in range(N_CH)]
    info = mne.create_info(ch_names, SFREQ, "mag")
    # Spread the sensors over a hemisphere so that topomaps can be drawn.
    idx = np.arange(N_CH) + 0.5
    polar = np.arccos(1 - idx / N_CH)
    azimuth = np.pi * (1 + np.sqrt(5)) * idx
    unit_pos = np.c_[
        np.sin(polar) * np.cos(azimuth),
        np.sin(polar) * np.sin(azimuth),
        np.cos(polar),
    ]
    for ch, this_pos in zip(info["chs"], 0.09 * unit_pos):
        ch["loc"][:3] = this_pos

    n_epochs = 2 * N_EPOCHS_PER_COND
    n_times = int(round((TMAX - TMIN) * SFREQ)) + 1
    times = TMIN + np.arange(n_times) / SFREQ
    data = rng.standard_normal((n_epochs, N_CH, n_times)) * UNIT
    event_id = np.r_[np.full(N_EPOCHS_PER_COND, 1), np.full(N_EPOCHS_PER_COND, 2)]
    # plant the effect in condition "b": a few channels, late in the epoch
    late = times >= EFFECT_TMIN - 1e-9
    data[N_EPOCHS_PER_COND:, :N_EFFECT_CH, late] += EFFECT_AMP * UNIT
    data = data * gain + offset
    events = np.c_[np.arange(n_epochs) * 100, np.zeros(n_epochs, int), event_id]
    # each group holds both conditions, as LOGO scoring requires
    grp = np.tile(np.arange(N_GROUPS), n_epochs // N_GROUPS)
    metadata = pd.DataFrame(dict(grp=grp))
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=dict(a=1, b=2),
        tmin=TMIN,
        metadata=metadata,
        verbose="error",
    )


def _run(
    tmp_path: Path, *, gain: float = 1.0, offset: float = 0.0, **cfg_overrides: Any
) -> dict[str, Any]:
    """Run the undecorated step function on synthetic epochs."""
    deriv_root = tmp_path / "derivatives"
    cfg = SimpleNamespace(
        conditions=dict(a="a", b="b"),
        contrasts=[("a", "b")],
        cov_rank=dict(tol=1e-4, tol_kind="relative"),
        decoding_which_epochs="cleaned",
        decoding_metric="roc_auc",
        decoding_epochs_tmin=TMIN,
        decoding_epochs_tmax=TMAX,
        decoding_n_splits=4,
        decoding_LOGO=False,
        decoding_LOGO_group=None,
        decoding_baseline=None,
        decoding_equalize=True,
        random_state=42,
        analyze_channels="ch_types",
        ch_types=["mag"],
        eeg_reference="average",
        acq=None,
        rec=None,
        space=None,
        datatype="meg",
        deriv_root=deriv_root,
        all_tasks=["test"],
    )
    for key, val in cfg_overrides.items():
        assert hasattr(cfg, key), key
        setattr(cfg, key, val)
    exec_params = SimpleNamespace(
        generate_reports=False,
        memory_file_method="mtime",
        deriv_root=deriv_root,
    )
    bids_path = BIDSPath(
        subject="01",
        task="test",
        processing="clean",
        suffix="epo",
        extension=".fif",
        datatype="meg",
        root=deriv_root,
        check=False,
    )
    bids_path.fpath.parent.mkdir(parents=True)
    _make_epochs(gain=gain, offset=offset).save(bids_path.fpath, verbose="error")
    kwargs: dict[str, Any] = dict(
        subject="01", session=None, task="test", condition1="a", condition2="b"
    )
    in_files = full_epochs.get_input_fnames_epochs_decoding(cfg=cfg, **kwargs)
    out_files: dict[str, Any] = full_epochs.run_epochs_decoding.__wrapped__(
        cfg=cfg, exec_params=exec_params, in_files=in_files, **kwargs
    )
    return out_files


def _load_scores(out_files: dict[str, Any]) -> np.ndarray:
    (key,) = (key for key in out_files if key.startswith("mat_"))
    return np.atleast_1d(loadmat(out_files[key][0])["scores"].squeeze())


def _load_weights(out_files: dict[str, Any]) -> pd.DataFrame:
    (key,) = (key for key in out_files if key.startswith("tsv_weights_"))
    return pd.read_csv(out_files[key][0], sep="\t")


@pytest.mark.parametrize(
    "cfg_overrides, n_scores",
    [
        pytest.param(dict(), 4, id="stratified"),
        pytest.param(
            dict(decoding_LOGO=True, decoding_LOGO_group="grp"),
            N_GROUPS,
            id="logo",
        ),
        pytest.param(
            dict(decoding_baseline=(-0.1, 0.0), decoding_equalize=False),
            4,
            id="baseline-no-equalize",
        ),
    ],
)
def test_full_epochs_decoding(
    tmp_path: Path, cfg_overrides: dict[str, Any], n_scores: int
) -> None:
    """Full epochs have to be vectorized before they reach the classifier."""
    out_files = _run(tmp_path, **cfg_overrides)

    scores = _load_scores(out_files)
    assert scores.shape == (n_scores,)
    assert scores.mean() > 0.8  # the planted effect is easy to decode

    # Patterns and filters come back in channel x time space
    weights = _load_weights(out_files)
    n_times = int(round((TMAX - TMIN) * SFREQ)) + 1
    assert len(weights) == 2 * N_CH * n_times
    assert set(weights["kind"]) == {"patterns", "filters"}
    assert weights["ch_name"].nunique() == N_CH
    assert weights["time"].nunique() == n_times
    assert np.isfinite(weights["value"]).all()

    # ... and the (channel, time) layout is the right way around: the pattern is
    # concentrated where the effect was planted.
    patterns = weights.query("kind == 'patterns'")
    in_effect = patterns["ch_name"].isin([f"MAG{ii:03d}" for ii in range(N_EFFECT_CH)])
    in_effect &= patterns["time"] >= EFFECT_TMIN - 1e-9
    values = patterns["value"].abs()
    assert values[in_effect].mean() > 2 * values[~in_effect].mean()


def test_full_epochs_decoding_too_few_epochs(tmp_path: Path) -> None:
    """Too few epochs should give NaN scores and no weights."""
    n_splits = N_EPOCHS_PER_COND + 6
    out_files = _run(tmp_path, decoding_n_splits=n_splits)

    scores = _load_scores(out_files)
    assert scores.shape == (n_splits,)
    assert np.isnan(scores).all()
    assert not any("weights" in key for key in out_files)
    assert not list((tmp_path / "derivatives").rglob("*weights*"))


def test_full_epochs_patterns_in_data_units(tmp_path: Path) -> None:
    """Patterns are mean-free covariance quantities in the units of the data."""
    ref = _load_weights(_run(tmp_path / "ref"))
    shifted = _load_weights(_run(tmp_path / "shifted", offset=5 * UNIT))
    scaled = _load_weights(_run(tmp_path / "scaled", gain=GAIN))
    index = ["kind", "ch_name", "time"]
    for other in (shifted, scaled):
        pd.testing.assert_frame_equal(other[index], ref[index])

    def patterns(df: pd.DataFrame) -> np.ndarray:
        return df.query("kind == 'patterns'")["value"].to_numpy()

    atol = 1e-5 * np.abs(patterns(ref)).max()
    # a constant offset of the data is not part of a pattern ...
    np.testing.assert_allclose(patterns(shifted), patterns(ref), rtol=0, atol=atol)
    # ... and a pattern scales with the data
    np.testing.assert_allclose(
        patterns(scaled), GAIN * patterns(ref), rtol=0, atol=GAIN * atol
    )
