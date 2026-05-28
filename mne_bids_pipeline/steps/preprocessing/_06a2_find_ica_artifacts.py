"""Find ICA artifacts.

This step automatically finds ECG- and EOG-related ICs in your data, and sets them
as bad components.

To actually remove designated ICA components from your data, you will have to
run the apply_ica step.
"""

import os
import pathlib
import shutil
from types import SimpleNamespace
from typing import Literal

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.utils import _pl
from mne_bids import BIDSPath

from mne_bids_pipeline._config_utils import (
    _bids_kwargs,
    _get_ss,
    get_eeg_reference,
    get_eog_channels,
    get_runs_tasks,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import get_parallel_backend, parallel_func
from mne_bids_pipeline._report import _agg_backend, _open_report
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import FloatArrayT, InFilesT, OutFilesT

N_JOBS=-1

# ---------------------------------------------------------------------------
# Per-component timeout (seconds) for plot_properties + savefig.
# Set to 0 or None to disable.
# ---------------------------------------------------------------------------
_ICA_PROPERTY_TIMEOUT: int | None = 120


def _get_ica_scratch_dir(ica_out_dir: pathlib.Path) -> pathlib.Path | None:
    """Return a node-local scratch directory for ICA figure I/O.

    Checks (in order): ICA_FIG_SCRATCH, SLURM_TMPDIR, TMPDIR, /tmp.
    Returns None only if ICA_FIG_SCRATCH is explicitly set to "0" or "off".
    """
    # Allow explicit opt-out
    explicit = os.environ.get("ICA_FIG_SCRATCH", "")
    if explicit.lower() in ("0", "off", "false", "no"):
        return None

    candidates = [
        os.environ.get("ICA_FIG_SCRATCH"),
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        "/tmp",
    ]
    for candidate in candidates:
        if candidate is not None:
            p = pathlib.Path(candidate)
            if p.is_dir() and os.access(p, os.W_OK):
                scratch = p / f"ica_figs_{os.getpid()}"
                scratch.mkdir(parents=True, exist_ok=True)
                return scratch
    return None


def _sync_ica_scratch_to_gpfs(
    scratch_dir: pathlib.Path,
    gpfs_dir: pathlib.Path,
) -> None:
    """Copy all files from *scratch_dir* to *gpfs_dir*, then clean up."""
    gpfs_dir.mkdir(parents=True, exist_ok=True)
    for src in scratch_dir.iterdir():
        if src.is_file():
            dst = gpfs_dir / src.name
            # Atomic-ish: write to tmp, then rename
            dst_tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
            try:
                shutil.copy2(src, dst_tmp)
                os.replace(dst_tmp, dst)
            except BaseException:
                dst_tmp.unlink(missing_ok=True)
                raise
    # Clean up scratch
    shutil.rmtree(scratch_dir, ignore_errors=True)


def _plot_property_with_timeout(
    ica,
    epochs,
    pick: int,
    bids_basename_for_figs,
    ica_out_dir: pathlib.Path,
    timeout: int | None,
):
    """Plot a single IC's properties and save to disk, with optional timeout.

    Uses signal.alarm (SIGALRM) for the timeout since this runs in the
    main process and avoids multiprocessing overhead.
    """
    import signal
    import matplotlib.pyplot as plt

    timed_out = False

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"IC {pick:03d} plot_properties timed out after {timeout}s")

    # Set alarm if timeout is enabled and we're on a POSIX system
    old_handler = None
    if timeout and hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)

    try:
        figs_props = ica.plot_properties(inst=epochs, picks=pick, show=False)
        if not isinstance(figs_props, list):
            figs_props = [figs_props]
        for fig in figs_props:
            suffix = f"icaProperties{pick:03d}"
            fig_path = _ica_fig_path(
                bids_basename_for_figs, ica_out_dir, "ica", suffix
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
    except TimeoutError:
        timed_out = True
        logger.warning(
            **gen_log_kwargs(
                message=f"⏱ IC {pick:03d}: timed out after {timeout}s, skipping."
            )
        )
    finally:
        # Cancel alarm and restore handler
        if timeout and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    return not timed_out


def _ica_fig_path(
    bids_basename: BIDSPath,
    ica_out_dir: pathlib.Path,
    processing: str,
    suffix: str,
    extension: str = ".png",
) -> pathlib.Path:
    """Build a BIDS-style filename in the ICA output directory."""
    bp = bids_basename.copy().update(
        processing=processing, suffix=suffix, extension=extension
    )
    return ica_out_dir / bp.basename


def detect_bad_components(
    *,
    cfg: SimpleNamespace,
    which: Literal["eog", "ecg"],
    epochs: mne.BaseEpochs | None,
    ica: mne.preprocessing.ICA,
    ch_names: list[str] | None,
    subject: str,
    session: str | None,
) -> tuple[list[int], FloatArrayT]:
    artifact = which.upper()
    if epochs is None:
        msg = (
            f"No {artifact} events could be found. "
            f"Not running {artifact} artifact detection."
        )
        logger.info(**gen_log_kwargs(message=msg))
        return [], np.zeros(0)
    msg = f"Performing automated {artifact} artifact detection …"
    logger.info(**gen_log_kwargs(message=msg))

    if which == "eog":
        inds, scores = ica.find_bads_eog(
            epochs,
            threshold=cfg.ica_eog_threshold,
            ch_name=ch_names,
        )
    else:
        inds, scores = ica.find_bads_ecg(
            epochs,
            method="ctps",
            threshold=cfg.ica_ecg_threshold,
            ch_name=ch_names,
        )

    if not inds:
        adjust_setting = f"ica_{which}_threshold"
        warn = (
            f"No {artifact}-related ICs detected, this is highly "
            f"suspicious. A manual check is suggested. You may wish to "
            f'lower "{adjust_setting}".'
        )
        logger.warning(**gen_log_kwargs(message=warn))
    else:
        msg = (
            f"Detected {len(inds)} {artifact}-related ICs in "
            f"{len(epochs)} {artifact} epochs: {', '.join([str(i) for i in inds])}"
        )
        logger.info(**gen_log_kwargs(message=msg))

    return inds, scores


def get_input_fnames_find_ica_artifacts(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> InFilesT:
    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=None,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
        extension=".fif",
    )
    in_files = dict()
    in_files["epochs"] = bids_basename.copy().update(processing="icafit", suffix="epo")
    _update_for_splits(in_files, "epochs", single=True)
    for run, task in cfg.runs_tasks:
        key = f"raw_task-{task}_run-{run}"
        in_files[key] = bids_basename.copy().update(
            run=run, task=task, processing=cfg.processing, suffix="raw"
        )
        _update_for_splits(in_files, key, single=True)
    in_files["ica"] = bids_basename.copy().update(
        processing="icafit", suffix="ica", task=None
    )
    return in_files


@failsafe_run(
    get_input_fnames=get_input_fnames_find_ica_artifacts,
)
def find_ica_artifacts(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    """Run ICA."""
    raw_fnames = [
        in_files.pop(f"raw_task-{task}_run-{run}") for run, task in cfg.runs_tasks
    ]
    bids_basename = (
        raw_fnames[0].copy().update(processing=None, split=None, run=None, task=None)
    )
    out_files = dict()
    out_files["ica"] = bids_basename.copy().update(processing="ica", suffix="ica")
    out_files["ecg"] = bids_basename.copy().update(processing="ica+ecg", suffix="ave")
    out_files["eog"] = bids_basename.copy().update(processing="ica+eog", suffix="ave")

    # DO NOT add this to out_files["ica"] because we expect it to be modified by users.
    # If the modify it and it's in out_files, caching will detect the hash change and
    # consider *this step* a cache miss, and it will run again, overwriting the user's
    # changes. Instead, we want the ica.apply step to rerun (which it will if the
    # file changes).
    out_files_components = bids_basename.copy().update(
        processing="ica", suffix="components", extension=".tsv"
    )

    # ICA figure/array output directory
    ica_out_dir = out_files["ica"].fpath.parent / "ICA"
    ica_out_dir.mkdir(exist_ok=True, parents=True)
    bids_basename_for_figs = bids_basename.copy()
    del bids_basename
    msg = "Loading ICA solution"
    logger.info(**gen_log_kwargs(message=msg))
    ica = mne.preprocessing.read_ica(in_files.pop("ica"))

    # Epochs used for ICA fitting
    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    # ECG component detection
    epochs_ecg = None
    ecg_ics: list[int] = []
    ecg_scores: FloatArrayT = np.zeros(0)
    if cfg.ica_use_ecg_detection:
        for ri, raw_fname in enumerate(raw_fnames):
            # Have the channels needed to make ECG epochs
            raw = mne.io.read_raw(raw_fname, preload=False)
            if cfg.ica_use_icalabel:
                raw.set_eeg_reference("average", projection=True).apply_proj()
            # ECG epochs
            if not (
                "ecg" in raw.get_channel_types()
                or "meg" in cfg.ch_types
                or "mag" in cfg.ch_types
            ):
                msg = (
                    "No ECG or magnetometer channels are present, cannot "
                    "automate artifact detection for ECG."
                )
                logger.info(**gen_log_kwargs(message=msg))
                break
            elif ri == 0:
                msg = "Creating ECG epochs …"
                logger.info(**gen_log_kwargs(message=msg))

                # We want to extract a total of 5 min of data for ECG epochs generation
                # (across all runs)
                total_ecg_dur = 5 * 60
                ecg_dur_per_run = total_ecg_dur / len(raw_fnames)
                t_mid = (raw.times[-1] + raw.times[0]) / 2
                raw = raw.crop(
                    tmin=max(t_mid - 1 / 2 * ecg_dur_per_run, 0),
                    tmax=min(t_mid + 1 / 2 * ecg_dur_per_run, raw.times[-1]),
                ).load_data()

                these_ecg_epochs = create_ecg_epochs(
                    raw,
                    baseline=(None, -0.2),
                    tmin=-0.5,
                    tmax=0.5,
                )
                del raw  # Free memory
                if len(these_ecg_epochs):
                    if epochs.reject is not None:
                        these_ecg_epochs.drop_bad(reject=epochs.reject)
                    if len(these_ecg_epochs):
                        if epochs_ecg is None:
                            epochs_ecg = these_ecg_epochs
                        else:
                            epochs_ecg = mne.concatenate_epochs(
                                [epochs_ecg, these_ecg_epochs], on_mismatch="warn"
                            )
                del these_ecg_epochs
            else:  # did not break so had usable channels
                ecg_ics, ecg_scores = detect_bad_components(
                    cfg=cfg,
                    which="ecg",
                    epochs=epochs_ecg,
                    ica=ica,
                    ch_names=None,  # we currently don't allow for custom channels
                    subject=subject,
                    session=session,
                )

    # EOG component detection

    # get subject and session specific EOG channel
    eog_chs_subj_sess = get_eog_channels(cfg.eog_channels, subject, session)

    if eog_chs_subj_sess is not None:
        # convert to list for type annotation compatibility
        eog_chs_subj_sess = list(eog_chs_subj_sess)

    epochs_eog = None
    eog_ics: list[int] = []
    eog_scores: FloatArrayT = np.zeros(0)
    if cfg.ica_use_eog_detection:
        for ri, raw_fname in enumerate(raw_fnames):
            raw = mne.io.read_raw_fif(raw_fname, preload=True)
            if cfg.ica_use_icalabel:
                raw.set_eeg_reference("average", projection=True).apply_proj()
            if eog_chs_subj_sess is not None:  # explicit None-check to allow []
                ch_names = eog_chs_subj_sess
                assert all([ch_name in raw.ch_names for ch_name in ch_names])
            else:
                eog_picks = mne.pick_types(raw.info, meg=False, eog=True)
                ch_names = [raw.ch_names[pick] for pick in eog_picks]
            if not ch_names:
                msg = "No EOG channel is present, cannot automate IC detection for EOG."
                logger.info(**gen_log_kwargs(message=msg))
                break
            elif ri == 0:
                msg = "Creating EOG epochs …"
                logger.info(**gen_log_kwargs(message=msg))
            these_eog_epochs = create_eog_epochs(
                raw,
                ch_name=ch_names,
                baseline=(None, -0.2),
            )
            if len(these_eog_epochs):
                if epochs.reject is not None:
                    these_eog_epochs.drop_bad(reject=epochs.reject)
                if len(these_eog_epochs):
                    if epochs_eog is None:
                        epochs_eog = these_eog_epochs
                    else:
                        epochs_eog = mne.concatenate_epochs(
                            [epochs_eog, these_eog_epochs], on_mismatch="warn"
                        )
        else:  # did not break
            eog_ics, eog_scores = detect_bad_components(
                cfg=cfg,
                which="eog",
                epochs=epochs_eog,
                ica=ica,
                ch_names=eog_chs_subj_sess,
                subject=subject,
                session=session,
            )

    # Run MNE-ICALabel if requested.
    exclude: list[int] = ecg_ics + eog_ics
    if cfg.ica_use_icalabel:
        icalabel_ics, icalabel_df, icalabel_labels, icalabel_report = _run_icalabel(
            cfg=cfg,
            ica=ica,
            epochs=epochs,
            mne_exclude=exclude,
            subject=subject,
            session=session,
        )
    else:
        icalabel_ics = []
        icalabel_df = pd.DataFrame()
        icalabel_labels = []
        icalabel_report = []

    exclude += icalabel_ics
    ica.exclude = sorted(set(exclude))
    del exclude

    # Save updated ICA to disk.
    # We also store the automatically identified ECG- and EOG-related ICs.
    msg = "Saving ICA solution and detected artifacts to disk."
    logger.info(**gen_log_kwargs(message=msg))
    ica.save(out_files["ica"], overwrite=True)

    # Create TSV.
    tsv_data = pd.DataFrame(
        dict(
            component=list(range(ica.n_components_)),
            type=["ica"] * ica.n_components_,
            description=["Independent Component"] * ica.n_components_,
            status=["good"] * ica.n_components_,
            status_description=["n/a"] * ica.n_components_,
        )
    )

    if cfg.ica_use_icalabel:
        for component, label in zip(icalabel_ics, icalabel_labels):
            row_idx = tsv_data["component"] == component
            tsv_data.loc[row_idx, "status"] = "bad"
            tsv_data.loc[row_idx, "status_description"] = (
                f"Auto-detected {label} (MNE-ICALabel)"
            )
    if cfg.ica_use_ecg_detection:
        for component in ecg_ics:
            row_idx = tsv_data["component"] == component
            tsv_data.loc[row_idx, "status"] = "bad"
            tsv_data.loc[row_idx, "status_description"] = (
                "Auto-detected ECG artifact (MNE)"
            )
    if cfg.ica_use_eog_detection:
        for component in eog_ics:
            row_idx = tsv_data["component"] == component
            tsv_data.loc[row_idx, "status"] = "bad"
            tsv_data.loc[row_idx, "status_description"] = (
                "Auto-detected EOG artifact (MNE)"
            )

    tsv_data.to_csv(out_files_components, sep="\t", index=False)

    # Lastly, add info about the epochs used for the ICA fit, and plot all ICs
    # for manual inspection.

    ecg_evoked = None if epochs_ecg is None else epochs_ecg.average()
    eog_evoked = None if epochs_eog is None else epochs_eog.average()

    # Save ECG and EOG evokeds to disk.
    for artifact_name, artifact_evoked in zip(("ecg", "eog"), (ecg_evoked, eog_evoked)):
        if artifact_evoked:
            msg = f"Saving {artifact_name.upper()} artifact: {out_files[artifact_name]}"
            logger.info(**gen_log_kwargs(message=msg))
            artifact_evoked.save(out_files[artifact_name], overwrite=True)
        else:
            # Don't track the non-existent output file
            del out_files[artifact_name]

    del artifact_name, artifact_evoked

    title = "ICA: components"
    tags = ("ica",)
    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
    ) as report:
        logger.info(**gen_log_kwargs(message=f'Adding "{title}" to report.'))
        report.add_ica(
            ica=ica,
            title=title,
            inst=epochs,
            ecg_evoked=ecg_evoked,
            eog_evoked=eog_evoked,
            ecg_scores=ecg_scores if len(ecg_scores) else None,
            eog_scores=eog_scores if len(eog_scores) else None,
            replace=True,
            n_jobs=1,  # avoid automatic parallelization
            tags=tags,
        )

    # Export ICA figures as PNGs and scores as NPY arrays
    logger.info(**gen_log_kwargs(message="Saving ICA figures to PNG files."))

    # Option 2: Use node-local scratch to avoid GPFS I/O stalls
    scratch_dir = _get_ica_scratch_dir(ica_out_dir)
    if scratch_dir is not None:
        fig_out_dir = scratch_dir
        logger.info(
            **gen_log_kwargs(
                message=f"Using local scratch for ICA figures: {scratch_dir}"
            )
        )
    else:
        fig_out_dir = ica_out_dir

    with _agg_backend():
        import matplotlib.pyplot as plt

        # --- Component topographies ---
        figs = ica.plot_components(colorbar=True, show=False)
        if not isinstance(figs, list):
            figs = [figs]
        for fi, fig in enumerate(figs):
            suffix = "icaComponents" if fi == 0 else f"icaComponents{fi + 1}"
            fig_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica", suffix
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- Component properties (with per-component timeout) ---
        if cfg.ica_plot_component_properties == "all":
            props_picks = list(range(ica.n_components_))
        else:  # "excluded"
            props_picks = list(ica.exclude)

        n_ok, n_timeout = 0, 0
        for pick in props_picks:
            ok = _plot_property_with_timeout(
                ica=ica,
                epochs=epochs,
                pick=pick,
                bids_basename_for_figs=bids_basename_for_figs,
                ica_out_dir=fig_out_dir,
                timeout=_ICA_PROPERTY_TIMEOUT,
            )
            if ok:
                n_ok += 1
            else:
                n_timeout += 1

        msg = (
            f"Component properties: {n_ok}/{len(props_picks)} saved"
            + (f" ({n_timeout} timed out)" if n_timeout else "")
        )
        logger.info(**gen_log_kwargs(message=msg))

        # save sensor names for matching to scores
        sensor_names_path = _ica_fig_path(
            bids_basename_for_figs, fig_out_dir, "ica", "sensorNames", ".tsv"
        )
        pd.DataFrame({"sensor": ica.ch_names}).to_csv(sensor_names_path, sep="\t", index=False)

        # --- IC sensor maps (mixing matrix topographies) ---
        ic_maps = ica.get_components()  # [n_channels, n_components]
        ic_labels = [f"IC{i:03d}" for i in range(ica.n_components_)]
        ic_maps_df = pd.DataFrame(
            ic_maps, columns=ic_labels, index=ica.ch_names
        )
        ic_maps_path = _ica_fig_path(
            bids_basename_for_figs, fig_out_dir, "ica", "icaSensorMaps", ".tsv"
        )
        ic_maps_df.to_csv(ic_maps_path, sep="\t", index_label="sensor")

        # Z-score sensor maps per IC (across channels) for weighted maps
        ic_std = ic_maps.std(axis=0, keepdims=True)
        ic_std[ic_std == 0] = 1  # avoid division by zero
        ic_maps_z = (ic_maps - ic_maps.mean(axis=0, keepdims=True)) / ic_std

        # --- ECG scores ---
        if len(ecg_scores) > 0:
            fig = ica.plot_scores(scores=ecg_scores, labels="ecg", show=False)
            fig_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica+ecg", "icaScores"
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Save ECG scores as TSV [n_IC, 1]
            ecg_scores_1d = ecg_scores.ravel()
            ecg_scores_df = pd.DataFrame(
                {"score": ecg_scores_1d}, index=ic_labels
            )
            ecg_scores_df.index.name = "component"
            tsv_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica+ecg", "icaScores", ".tsv"
            )
            ecg_scores_df.to_csv(tsv_path, sep="\t")

            # Save score-weighted z-scored sensor maps [n_channels, n_IC]
            ecg_weighted = ic_maps_z * ecg_scores_1d[np.newaxis, :]
            ecg_weighted_df = pd.DataFrame(
                ecg_weighted, columns=ic_labels, index=ica.ch_names
            )
            weighted_path = _ica_fig_path(
                bids_basename_for_figs,
                fig_out_dir,
                "ica+ecg",
                "icaWeightedMaps",
                ".tsv",
            )
            ecg_weighted_df.to_csv(weighted_path, sep="\t", index_label="sensor")

        # --- EOG scores ---
        if len(eog_scores) > 0:
            fig = ica.plot_scores(scores=eog_scores, labels="eog", show=False)
            fig_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica+eog", "icaScores"
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Save EOG scores as TSV [n_IC, 1]
            # If multi-channel EOG, pick the channel with max abs correlation
            # find_bads_eog returns a list of arrays (one per channel) when
            # ch_name is a list; convert to ndarray so .ndim works correctly.
            eog_scores = np.array(eog_scores)
            if eog_scores.ndim > 1:
                best_ch = np.argmax(np.abs(eog_scores), axis=0, keepdims=True)
                eog_scores_1d = np.take_along_axis(
                    eog_scores, best_ch, axis=0
                ).ravel()
            else:
                eog_scores_1d = eog_scores
            eog_scores_df = pd.DataFrame(
                {"score": eog_scores_1d}, index=ic_labels
            )
            eog_scores_df.index.name = "component"
            tsv_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica+eog", "icaScores", ".tsv"
            )
            eog_scores_df.to_csv(tsv_path, sep="\t")

            # Save score-weighted z-scored sensor maps [n_channels, n_IC]
            eog_weighted = ic_maps_z * eog_scores_1d[np.newaxis, :]
            eog_weighted_df = pd.DataFrame(
                eog_weighted, columns=ic_labels, index=ica.ch_names
            )
            weighted_path = _ica_fig_path(
                bids_basename_for_figs,
                fig_out_dir,
                "ica+eog",
                "icaWeightedMaps",
                ".tsv",
            )
            eog_weighted_df.to_csv(weighted_path, sep="\t", index_label="sensor")

        # --- ECG sources ---
        if ecg_evoked is not None:
            fig = ica.plot_sources(inst=ecg_evoked, show=False)
            fig_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica+ecg", "icaSources"
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- EOG sources ---
        if eog_evoked is not None:
            fig = ica.plot_sources(inst=eog_evoked, show=False)
            fig_path = _ica_fig_path(
                bids_basename_for_figs, fig_out_dir, "ica+eog", "icaSources"
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- ICA overlay (original vs cleaned signal) ---
        fig = ica.plot_overlay(inst=epochs.average(), show=False, on_baseline="reapply")
        fig_path = _ica_fig_path(
            bids_basename_for_figs, fig_out_dir, "ica", "icaOverlay"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Sync local scratch files to GPFS (single batch copy)
    if scratch_dir is not None:
        logger.info(
            **gen_log_kwargs(message="Syncing ICA figures from local scratch to GPFS.")
        )
        _sync_ica_scratch_to_gpfs(scratch_dir, ica_out_dir)
        logger.info(**gen_log_kwargs(message="ICA figure sync complete."))

        if cfg.ica_use_icalabel:
            _add_report_icalabel(
                report=report,
                ica=ica,
                icalabel_report=icalabel_report,
                icalabel_df=icalabel_df,
                tags=tags,
                subject=subject,
                session=session,
            )

    msg = 'Carefully review the extracted ICs and mark components "bad" in:'
    logger.info(**gen_log_kwargs(message=msg, emoji="🛑"))
    logger.info(**gen_log_kwargs(message=str(out_files_components), emoji="🛑"))

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


_ICALABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]


def _run_icalabel(
    *,
    cfg: SimpleNamespace,
    ica: mne.preprocessing.ICA,
    epochs: mne.BaseEpochs,
    mne_exclude: list[int],
    subject: str,
    session: str | None,
) -> tuple[list[int], pd.DataFrame, list[str], list[tuple[str, float, bool]]]:
    from mne_icalabel.iclabel import iclabel_label_components

    icalabel_ics: list[int] = []
    icalabel_labels: list[str] = []
    icalabel_report: list[tuple[str, float, bool]] = []
    msg = "Performing automated artifact detection (MNE-ICALabel) …"
    logger.info(**gen_log_kwargs(message=msg))

    icalabel_class_probabilities = iclabel_label_components(
        inst=epochs, ica=ica, inplace=False
    )
    icalabel_component_labels = [
        _ICALABEL_CLASSES[i] for i in icalabel_class_probabilities.argmax(axis=1)
    ]
    icalabel_max_probabilities = icalabel_class_probabilities.max(axis=1)

    # logic for exclusion of components - look at each component
    for idx, (label, prob) in enumerate(
        zip(icalabel_component_labels, icalabel_max_probabilities)
    ):
        # get its probability table over all classes e.g.
        # [0.7, 0.1, 0.1, 0.05, 0.05, 0, 0]
        probs = dict(zip(_ICALABEL_CLASSES, icalabel_class_probabilities[idx]))

        exclude_component = any(  # ONLY
            # IF the component looked at does have a probability higher than the
            # exclusion_threshold in any of the NOT included classes
            cls not in cfg.ica_icalabel_include
            and p >= cfg.ica_exclusion_thresholds.get(cls, 0.8)
            for cls, p in probs.items()
        ) and not any(  # AND
            # IF the component looked at does NOT have a probability higher than the
            # class_threshold in any of the included classes
            cls in cfg.ica_icalabel_include
            and p >= cfg.ica_class_thresholds.get(cls, 0.3)
            for cls, p in probs.items()
        )

        if exclude_component:
            # THEN exclude that component
            icalabel_ics.append(idx)
            icalabel_labels.append(label)
            icalabel_report.append((label, prob, True))
        else:
            # ELSE keep it
            icalabel_report.append((label, prob, False))

    msg = (
        f"Detected {len(icalabel_ics)} artifact-related independent "
        f"component{_pl(icalabel_ics)} in {len(epochs)} epochs."
    )
    logger.info(**gen_log_kwargs(message=msg))
    icalabel_df = pd.DataFrame(icalabel_class_probabilities, columns=_ICALABEL_CLASSES)

    icalabel_df["Component"] = [
        f"ICA{i:03d}" for i in range(len(icalabel_component_labels))
    ]
    icalabel_df["PredictedLabel"] = icalabel_component_labels
    icalabel_df["MaxProbability"] = icalabel_max_probabilities
    exclude = mne_exclude + icalabel_ics
    icalabel_df["Excluded"] = [
        i in exclude for i in range(len(icalabel_component_labels))
    ]
    return icalabel_ics, icalabel_df, icalabel_labels, icalabel_report


def _add_report_icalabel(
    *,
    report: mne.Report,
    ica: mne.preprocessing.ICA,
    icalabel_report: list[tuple[str, float, bool]],
    icalabel_df: pd.DataFrame,
    tags: tuple[str, ...],
    subject: str | None = None,
    session: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    section = "ICA: ICALabel"
    logger.info(**gen_log_kwargs(message=f'Adding "{section}" to report.'))

    icalabel_prob_table_html = (
        """
    <table border="1" cellspacing="0" cellpadding="5" style="border-collapse:collapse; text-align:center; font-size:13px; width:100%;">
    <thead>
    <tr style="background-color:#eee;">
    <th>Component</th><th>Predicted Label</th><th>Max Prob</th><th>Excluded</th>
    """  # noqa: E501
        + "".join(f"<th>{cls}</th>" for cls in _ICALABEL_CLASSES)
        + "</tr></thead><tbody>"
    )
    for _, row in icalabel_df.iterrows():
        bg_color = "#FFB3B3" if row.Excluded else "#B3B3FF"
        text_color = "color:black;"
        prob_cells = "".join(f"<td>{row[c]:0.3f}</td>" for c in _ICALABEL_CLASSES)
        icalabel_prob_table_html += (
            f"<tr style='background-color:{bg_color};{text_color}'>"
            f"<td>{row.Component}</td>"
            f"<td>{row.PredictedLabel}</td>"
            f"<td>{row.MaxProbability:0.3f}</td>"
            f"<td>{'Yes' if row.Excluded else 'No'}</td>"
            f"{prob_cells}</tr>\n"
        )
    icalabel_prob_table_html += "</tbody></table>"
    report.add_html(
        title="ICALabel: report",
        html=icalabel_prob_table_html,
        tags=tags,
        section=section,
    )

    icalabel_map: dict[str, list[int]] = {}
    for i, (label, _, _) in enumerate(icalabel_report):
        icalabel_map.setdefault(label, []).append(i)

    for label, indices in icalabel_map.items():
        n_col = 4
        n_row = (len(indices) - 1) // n_col + 1
        fig, axes = plt.subplots(
            n_row,
            n_col,
            figsize=(4 * n_col, 3 * n_row),
            layout="constrained",
        )
        axes = axes.flatten()
        for j, ic in enumerate(indices):
            prob = icalabel_report[ic][1]
            status = "excluded" if ic in ica.exclude else "included"
            fcolor = "red" if ic in ica.exclude else "blue"
            ica.plot_components(picks=ic, axes=[axes[j]], show=False)
            axes[j].text(
                0.5,
                -0.15,
                f"ICA{ic:03d} — {label}, {prob:.3f} ({status})",
                ha="center",
                va="top",
                fontsize=8,
                transform=axes[j].transAxes,
                bbox=dict(facecolor=fcolor, alpha=0.3, pad=4),
            )
        for ax in axes[len(indices) :]:
            fig.delaxes(ax)
        report.add_figure(
            fig=fig,
            title=f"ICALabel: {label} components",
            section=section,
            tags=tags + (label.replace(" ", "_").replace("-", "_"),),
        )
        plt.close(fig)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        runs_tasks=get_runs_tasks(config=config, subject=subject, session=session),
        ica_use_eog_detection=config.ica_use_eog_detection,
        ica_eog_threshold=config.ica_eog_threshold,
        ica_use_ecg_detection=config.ica_use_ecg_detection,
        ica_ecg_threshold=config.ica_ecg_threshold,
        ica_use_icalabel=config.ica_use_icalabel,
        ica_icalabel_include=config.ica_icalabel_include,
        ica_exclusion_thresholds=config.ica_exclusion_thresholds,
        ica_class_thresholds=config.ica_class_thresholds,
        ica_plot_component_properties=config.ica_plot_component_properties,
        ch_types=config.ch_types,
        eeg_reference=get_eeg_reference(config),
        eog_channels=config.eog_channels,
        processing="filt" if config.regress_artifact is None else "regress",
        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run ICA."""
    if config.spatial_filter != "ica":
        logger.info(**gen_log_kwargs(message="SKIP"))
        return

    ss = _get_ss(config=config)
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            find_ica_artifacts,
            exec_params=config.exec_params,
            n_iter=len(ss),
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject, session=session),
                exec_params=config.exec_params,
                subject=subject,
                session=session,
            )
            for subject, session in ss
        )
    save_logs(config=config, logs=logs)
