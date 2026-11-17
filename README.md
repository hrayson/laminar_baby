# Laminar resolution in infant MEG

Analysis and simulation code for estimating whether infant MEG can distinguish event-related fields generated at different cortical depths.

## Conference contribution

Rayson, H., et al. (2024). [Laminar resolution in infant MEG: Simulating event-related fields for non-invasive cortical layer inference](https://hal.science/hal-04978270). Poster presented at BioMag 2024, Sydney. [Poster PDF](https://hal.science/hal-04978270/document)

## Code

- [`preprocess.ipynb`](preprocess.ipynb) and [`process_surfaces.ipynb`](process_surfaces.ipynb) prepare MEG and anatomical inputs.
- [`simulate.ipynb`](simulate.ipynb) and [`inversion.ipynb`](inversion.ipynb) run simulations and source inversion.
- [`run_snr_simulation.py`](run_snr_simulation.py) and [`run_coreg_err_simulation.py`](run_coreg_err_simulation.py) test signal-to-noise ratio and coregistration error.

Set the dataset and output locations in [`settings.json`](settings.json) for your environment. The notebooks expect local infant MEG and anatomical derivatives.

See [LICENSE](LICENSE) for the code licence.
