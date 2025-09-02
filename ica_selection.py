import os

import mne
import matplotlib.pyplot as plt

deriv_dir = '/home/bonaiuto/laminar_baby/data/derivatives/P001_T1'
raw_path = os.path.join(deriv_dir, 'sss_p001_t1_run01_raw.fif')
event_path = os.path.join(deriv_dir, 'p001_t1_run01_raw-eve.fif')
ica_path = os.path.join(deriv_dir, 'ica_p001_t1_run01_raw.fif')

raw = mne.io.read_raw_fif(
    raw_path, preload=True, verbose=False
)

events = mne.read_events(event_path)

ica = mne.preprocessing.read_ica(
    ica_path, verbose=False
)

raw_filtered = raw.copy()
raw_filtered.load_data().crop(
    tmin=raw_filtered.times[events[0, 0]]
)
raw_filtered.filter(
    l_freq=1.,
    h_freq=20,
    n_jobs=-1
)
ica.plot_components(inst=raw_filtered, show=False)

ica.plot_sources(inst=raw_filtered, show=False)

plt.show()