"""Faces dataset."""

bids_root = "~/mne_data/ds000117"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds000117"

task = "facerecognition"
ch_types = ["meg"]
runs = ["01", "02"]
sessions = ["meg"]
subjects = ["01"]

raw_resample_sfreq = 125.0
crop_runs = (0, 300)  # Reduce memory usage on CI system

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
process_empty_room = True

mf_reference_run = "02"
mf_cal_fname = bids_root + "/derivatives/meg_derivatives/sss_cal.dat"
mf_ctc_fname = bids_root + "/derivatives/meg_derivatives/ct_sparse.fif"
mf_int_order = 9
mf_ext_order = 2

reject = {"grad": 4000e-13, "mag": 4e-12}
conditions = ["Famous", "Unfamiliar", "Scrambled"]
contrasts = [
    ("Famous", "Scrambled"),
    ("Unfamiliar", "Scrambled"),
    ("Famous", "Unfamiliar"),
]

decode = True
decoding_time_generalization = True

run_source_estimation = False
