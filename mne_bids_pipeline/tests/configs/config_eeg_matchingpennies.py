"""Matchingpennies EEG experiment."""

bids_root = "~/mne_data/eeg_matchingpennies"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/eeg_matchingpennies"

subjects = ["05"]
task = "matchingpennies"
ch_types = ["eeg"]
interactive = False
reject = {"eeg": 150e-6}
conditions = ["raised-left", "raised-right"]
contrasts = [("raised-left", "raised-right")]
decode = True

interpolate_bads_grand_average = False

l_freq = None
h_freq = 100
zapline_fline = 50
zapline_iter = False
