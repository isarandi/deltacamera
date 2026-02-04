#!/usr/bin/env python3
"""
Scripts used to extract camera calibrations from various datasets.
Each function can be run standalone or via the CLI at the bottom.

The exact code used to create the .npz files in this directory.
"""

import numpy as np


def extract_dna_rendering(data_root, output_path):
    """Extract Brown-Conrady calibrations from DNA-Rendering dataset.

    Original command:
    .venv/bin/python << 'EOF'
    ...
    EOF
    """
    import h5py

    # Open one SMC file to extract calibrations
    smc_path = f'{data_root}/dna_rendering/data_used_in_4K4D/annotations/0012_11_annots.smc'
    smc = h5py.File(smc_path, 'r')

    D_list = []
    K_list = []
    cam_types = []

    for cam_id in sorted(smc['Camera_Parameter'].keys(), key=int):
        cam = smc['Camera_Parameter'][cam_id]
        D_list.append(cam['D'][()])
        K_list.append(cam['K'][()])
        cam_types.append('5mp' if int(cam_id) < 48 else '12mp')

    smc.close()

    D_all = np.array(D_list, dtype=np.float32)
    K_all = np.array(K_list, dtype=np.float32)

    # Select a diverse subset (10-15 cameras) for test data
    # Pick: min/max k1, min/max k2, some 5mp, some 12mp, some middle-of-road
    k1 = D_all[:, 0]
    k2 = D_all[:, 1]

    selected_indices = list(set([
        int(np.argmin(k1)),  # Most negative k1 (strong barrel)
        int(np.argmax(k1)),  # Most positive k1 (pincushion)
        int(np.argmin(k2)),  # Extreme k2
        int(np.argmax(k2)),  # Extreme k2 other direction
        0, 10, 20, 30, 40,   # Sample across 5mp range
        48, 52, 56,          # Sample from 12mp
    ]))
    selected_indices = sorted(selected_indices)

    print(f"Selected {len(selected_indices)} cameras: {selected_indices}")

    # Save to npz
    np.savez(
        output_path,
        distortion_coeffs=D_all[selected_indices],
        intrinsic_matrices=K_all[selected_indices],
        camera_indices=np.array(selected_indices),
        camera_types=np.array([cam_types[i] for i in selected_indices]),
    )

    print(f"\nSaved to {output_path}")


def extract_egohumans(data_root, output_path):
    """Extract fisheye calibrations from EgoHumans dataset.

    Original command:
    .venv/bin/python << 'EOF'
    ...
    EOF
    """
    import glob

    # Find all cameras.txt files
    camera_files = sorted(glob.glob(f'{data_root}/egohumans/*/*/colmap/workplace/cameras.txt'))

    all_fisheye = []
    all_intrinsics = []
    all_resolutions = []
    cam_types = []

    # Read from first file (they're all the same cameras)
    with open(camera_files[0]) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    for line in lines:
        parts = line.split()
        cam_id = int(parts[0])
        w, h = int(parts[2]), int(parts[3])
        fx, fy = float(parts[4]), float(parts[5])
        cx, cy = float(parts[6]), float(parts[7])
        k1, k2, k3, k4 = [float(x) for x in parts[8:12]]

        all_fisheye.append([k1, k2, k3, k4])
        all_intrinsics.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        all_resolutions.append([w, h])
        cam_types.append('aria_ego' if cam_id <= 4 else 'exo')

    all_fisheye = np.array(all_fisheye, dtype=np.float32)
    all_intrinsics = np.array(all_intrinsics, dtype=np.float32)
    all_resolutions = np.array(all_resolutions, dtype=np.int32)
    cam_types = np.array(cam_types)

    # Save to test data file
    np.savez(
        output_path,
        distortion_coeffs=all_fisheye,
        intrinsic_matrices=all_intrinsics,
        resolutions=all_resolutions,
        camera_types=cam_types,
    )

    print(f"Saved {len(all_fisheye)} fisheye calibrations to {output_path}")


def extract_jrdb(data_root, output_path):
    """Extract Brown-Conrady calibrations from JRDB dataset.

    Original command:
    .venv/bin/python << 'EOF'
    ...
    EOF
    """
    import re

    with open(f'{data_root}/jrdb/train_dataset/calibration/cameras.yaml') as f:
        content = f.read()

    # Split by sensor_X:
    sensor_blocks = re.split(r'\n  sensor_', content)[1:]  # Skip first part before any sensor

    distortion_coeffs = []
    intrinsic_matrices = []
    camera_names = []
    resolutions = []

    for block in sensor_blocks:
        lines = block.strip().split('\n')
        sensor_id = int(lines[0].rstrip(':'))

        # Find D, K, width, height
        width = height = None
        D = None
        K_lines = []
        in_K = False

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('width:'):
                width = int(line.split(':')[1].strip())
            elif line.startswith('height:'):
                height = int(line.split(':')[1].strip())
            elif line.startswith('D:'):
                D = np.array(list(map(float, line.split(':')[1].strip().split())), dtype=np.float32)
            elif line.startswith('K:'):
                in_K = True
            elif in_K and not line.startswith('#') and not line.startswith('R:'):
                if re.match(r'^[\d\.\-e\s]+$', line):
                    K_lines.append(list(map(float, line.split())))
                    if len(K_lines) == 3:
                        in_K = False
            elif line.startswith('R:'):
                in_K = False

        if D is not None and len(K_lines) == 3:
            K = np.array(K_lines, dtype=np.float32)
            distortion_coeffs.append(D)
            intrinsic_matrices.append(K)
            camera_names.append(f'sensor_{sensor_id}')
            resolutions.append([width, height])
            print(f"sensor_{sensor_id}: D={D}, fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    print(f"\nTotal cameras: {len(distortion_coeffs)}")

    # Save
    np.savez(
        output_path,
        distortion_coeffs=np.array(distortion_coeffs),
        intrinsic_matrices=np.array(intrinsic_matrices),
        camera_names=np.array(camera_names),
        resolutions=np.array(resolutions, dtype=np.int32),
    )
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Extract camera calibrations from datasets')
    parser.add_argument('--dataset', required=True,
                        choices=['dna_rendering', 'egohumans', 'jrdb'])
    parser.add_argument('--data-root', required=True,
                        help='Root directory containing the dataset')
    parser.add_argument('--output', help='Output path (default: in tests/data/)')
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    if args.output:
        output_path = args.output
    elif args.dataset == 'dna_rendering':
        output_path = output_dir / 'dna_rendering_calibrations.npz'
    elif args.dataset == 'egohumans':
        output_path = output_dir / 'egohumans_fisheye_calibrations.npz'
    elif args.dataset == 'jrdb':
        output_path = output_dir / 'jrdb_calibrations.npz'

    if args.dataset == 'dna_rendering':
        extract_dna_rendering(args.data_root, str(output_path))
    elif args.dataset == 'egohumans':
        extract_egohumans(args.data_root, str(output_path))
    elif args.dataset == 'jrdb':
        extract_jrdb(args.data_root, str(output_path))
