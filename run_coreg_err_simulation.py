import glob
import json
import os
import shutil
import pickle
import sys
import time

import numpy as np
import nibabel as nib
from scipy.spatial.transform import Rotation as R

from lameg.invert import coregister, invert_ebb
from lameg.simulate import run_current_density_simulation
from lameg.laminar import model_comparison
import spm_standalone


def run(sim_vertex, err_level, json_file):
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    out_path = os.path.join(parameters['output_path'], 'coreg_err_simulations')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    output_file = os.path.join(
        out_path,
        f"vx_{sim_vertex}_err_{err_level}.pickle"
    )

    if not os.path.exists(output_file):
        nas = [-12.4, 86.4, -35.4]
        lpa = [-49.4, 5.0, -41.7]
        rpa = [51.8, 13.6, -41.7]

        orig_fid = np.array([np.array(nas), np.array(lpa), np.array(rpa)])
        mean_fid = np.mean(orig_fid, axis=0)
        zero_mean_fid = np.hstack([(orig_fid - mean_fid), np.ones((3, 1))])

        spm = spm_standalone.initialize()

        # Where to put simulated data
        tmp_dir = os.path.join(out_path, f'coreg_err_vx_{sim_vertex}_coreg_err_{err_level}')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m/T1.nii'),
                    os.path.join(tmp_dir, 'T1.nii'))
        surf_files = glob.glob(os.path.join(parameters['dataset_path'], 'derivatives/217/6m/T1*.gii'))
        for surf_file in surf_files:
            shutil.copy(surf_file, tmp_dir)
        mri_fname = os.path.join(tmp_dir, 'T1.nii')

        shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m/multilayer.11.ds.link_vector.fixed.gii'),
                    os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed.gii'))
        multilayer_mesh_fname = os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed.gii')
        shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m/multilayer.11.ds.link_vector.fixed_warped.gii'),
                    os.path.join(tmp_dir, 'multilayer.11.ds.link_vector.fixed_warped.gii'))
        shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m/FWHM5.00_multilayer.11.ds.link_vector.fixed.mat'),
                    os.path.join(tmp_dir, 'FWHM5.00_multilayer.11.ds.link_vector.fixed.mat'))

        # Load multilayer mesh and compute the number of vertices per layer
        mesh = nib.load(multilayer_mesh_fname)
        n_layers = 11
        verts_per_surf = int(mesh.darrays[0].data.shape[0] / n_layers)

        # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used for the source
        # reconstruction
        orientation_method='link_vector.fixed'
        layers = np.linspace(1, 0, n_layers)
        layer_fnames = []
        for layer in layers:

            if layer == 1:
                name = f'pial.ds.{orientation_method}'
            elif layer == 0:
                name = f'white.ds.{orientation_method}'
            else:
                name = f'{layer:.3f}.ds.{orientation_method}'

            shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m', f'{name}.gii'),
                        os.path.join(tmp_dir, f'{name}.gii'))
            layer_fnames.append(os.path.join(tmp_dir, f'{name}.gii'))
            shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m', f'{name}_warped.gii'),
                        os.path.join(tmp_dir, f'{name}_warped.gii'))
            shutil.copy(os.path.join(parameters['dataset_path'], 'derivatives/217/6m', f'FWHM5.00_{name}.mat'),
                        os.path.join(tmp_dir, f'FWHM5.00_{name}.mat'))

        # Extract base name and path of data file
        data_file = os.path.join(parameters['dataset_path'], 'derivatives/P001_T1/Pspm_p001_t1_run01_epo.mat')
        data_path, data_file_name = os.path.split(data_file)
        data_base = os.path.splitext(data_file_name)[0]

        # Copy data files to tmp directory
        shutil.copy(
            os.path.join(data_path, f'{data_base}.mat'),
            os.path.join(tmp_dir, f'{data_base}.mat')
        )
        shutil.copy(
            os.path.join(data_path, f'{data_base}.dat'),
            os.path.join(tmp_dir, f'{data_base}.dat')
        )

        # Construct base file name for simulations
        base_fname = os.path.join(tmp_dir, f'{data_base}.mat')

        # Size of simulated patch of activity (mm)
        sim_patch_size = 5
        SNR = 0

        # Patch size to use for inversion (in this case it matches the simulated patch size)
        patch_size = 5
        # Number of temporal modes to use for EBB inversion
        n_temp_modes = 4

        # Coregister data to multilayer mesh
        coregister(
            nas,
            lpa,
            rpa,
            mri_fname,
            multilayer_mesh_fname,
            base_fname,
            fid_labels=['Nasion', 'LPA', 'RPA'],
            viz=False,
            spm_instance=spm
        )

        # Run inversion
        [_, _] = invert_ebb(
            multilayer_mesh_fname,
            base_fname,
            n_layers,
            patch_size=patch_size,
            n_temp_modes=n_temp_modes,
            viz=False,
            spm_instance=spm
        )

        # Frequency of simulated sinusoid (Hz)
        freq = 20
        # Strength of simulated activity (nAm)
        dipole_moment = 10
        # Sampling rate (must match the data file)
        s_rate = 2000

        # Generate 1s of a sine wave at a sampling rate of 600Hz (to match the data file)
        time = np.linspace(-.5, 1.5, s_rate + 1)
        sim_signal = np.sin(time * freq * 2 * np.pi).reshape(1, -1)

        # Now simulate at the corresponding vertex on each layer, and for each simulation, run model comparison across
        # all layers
        all_layerF = []

        for l in range(n_layers):
            print(f'Simulating in layer {l}')
            l_vertex = l * verts_per_surf + sim_vertex
            prefix = f'sim_{sim_vertex}_{l}_'

            l_sim_fname = run_current_density_simulation(
                base_fname,
                prefix,
                l_vertex,
                sim_signal,
                dipole_moment,
                sim_patch_size,
                SNR,
                spm_instance=spm
            )

            for i in range(1000):
                # Translation vector
                translation = err_level
                shift_vec = np.random.randn(3)
                shift_vec = shift_vec / np.linalg.norm(shift_vec) * np.random.randn() * translation

                # Rotation vector
                rotation_rad = err_level * np.pi / 180.0
                rot_vec = np.random.randn(3)
                rot_vec = rot_vec / np.linalg.norm(rot_vec) * np.random.randn() * rotation_rad

                # Apply transformation to fiducial locations
                P = np.concatenate((shift_vec, rot_vec))
                rotation = R.from_rotvec(P[3:])
                rotation_matrix = rotation.as_matrix()
                A = np.eye(4)
                A[:3, :3] = rotation_matrix
                A[:3, 3] = P[:3]

                # Transform zero_mean_fid
                new_fid_homogeneous = (A @ zero_mean_fid.T).T
                new_fid = new_fid_homogeneous[:, :3] + mean_fid
                max_dist = np.max(np.sqrt(np.sum((new_fid - orig_fid) ** 2, axis=-1)))
                if np.abs(err_level - max_dist) < .05:
                    break

            [layerF, _] = model_comparison(
                new_fid[0,:],
                new_fid[1,:],
                new_fid[2,:],
                mri_fname,
                layer_fnames,
                l_sim_fname,
                viz=False,
                spm_instance=spm,
                coregister_kwargs={
                    'fid_labels': ['Nasion', 'LPA', 'RPA']
                },
                invert_kwargs={
                    'patch_size': patch_size,
                    'n_temp_modes': n_temp_modes
                }
            )
            all_layerF.append(layerF)
        all_layerF = np.array(all_layerF)

        sim_vx_res = {
            'sim_vertex': sim_vertex,
            'err_level': err_level,
            'all_layerF': all_layerF,
            'new_fid': new_fid
        }

        with open(output_file, "wb") as fp:
            pickle.dump(sim_vx_res, fp)

        shutil.rmtree(tmp_dir)

        spm.terminate()


if __name__=='__main__':
    n_vertices = 36262
    np.random.seed(42)
    vertices = np.random.randint(0, n_vertices, 50)
    err_levels=[0, 0.5, 1, 2, 3, 4, 5]

    all_verts = []
    all_err_levels = []
    for vert in vertices:
        for err in err_levels:
            all_verts.append(vert)
            all_err_levels.append(err)
    np.random.seed(int(time.time()))

    # parsing command line arguments
    try:
        sim_idx = int(sys.argv[1])
    except:
        print("incorrect simulation index")
        sys.exit()

    try:
        json_file = sys.argv[2]
        print("USING:", json_file)
    except:
        json_file = "settings.json"
        print("USING:", json_file)

    vertex_idx = all_verts[sim_idx]
    err_level = all_err_levels[sim_idx]

    run(vertex_idx, err_level, json_file)