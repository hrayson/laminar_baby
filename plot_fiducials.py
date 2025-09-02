import os
import numpy as np
import mne
import matplotlib.pyplot as plt

raw_dir='/home/bonaiuto/laminar_baby/data/raw/P001_T1/'
deriv_dir='/home/bonaiuto/laminar_baby/data/derivatives/P001_T1/'

raw_path = os.path.join(raw_dir, 'p001_t1_run01-raw.fif')
raw = mne.io.read_raw_fif(raw_path, verbose=False, allow_maxshield=True)

from mne.viz import plot_alignment
fig = plot_alignment(raw.info, meg=("helmet", "sensors"), eeg=False, coord_frame="meg", show_axes=True, verbose=True,
                     dig=True)
plt.show()

# load and process head position from cHPI
chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=raw.info)
print(f"cHPI coil frequencies extracted from raw: {chpi_freqs} Hz")

chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose=True)
chpi_locs       = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes, verbose=True)
head_pos        = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)

used_coils=np.array([0, 2, 3])


# Selecting the positions of all coils at the first time point
first_time_point_positions = chpi_locs['rrs'][0]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extracting x, y, z coordinates for each coil
x = first_time_point_positions[used_coils, 0]
y = first_time_point_positions[used_coils, 1]
z = first_time_point_positions[used_coils, 2]

ax.scatter(x, y, z)

# Labeling the points (optional, for clarity)
for i, (x_val, y_val, z_val) in enumerate(zip(x, y, z)):
    ax.text(x_val, y_val, z_val, f'Coil {i+1}', color='red')

ax.set_xlim([-.06, .06])
ax.set_ylim([-.04, .08])
ax.set_zlim([-.09, .03])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Positions of Fiducial Coils at First Time Point')

plt.show()
