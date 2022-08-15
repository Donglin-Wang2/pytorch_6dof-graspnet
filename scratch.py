from utils.visualization_utils import *
import grasp_estimator

import glob
import os
import argparse

import numpy as np
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation as R


def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='./checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='./checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    # parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser

def get_min_max_center(pcd):
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    for pnt in pcd:
        aggVertices += pnt
        numVertices += 1
        minVertex = np.minimum(pnt, minVertex)
        maxVertex = np.maximum(pnt, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info


def normalize_points(pcd):
    result = []
    stats = get_min_max_center(pcd)
    diag = np.array(stats['max']) - np.array(stats['min'])
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    for v in pcd:
        v_new = (v - c) * norm
        result.append(v_new)
    return np.stack(result), norm, c

def normalize_points2(pcd):
    result = []
    stats = get_min_max_center(pcd)
    c = stats['centroid']
    for v in pcd:
        v_new = (v - c) 
        result.append(v_new)
    return np.stack(result)



if __name__ == '__main__':
    for npy_file in glob.glob(os.path.join('./demo/data', '*.npy')):
        data = np.load(npy_file, allow_pickle=True,
                        encoding="latin1").item()
        for k, v in data.items():
            if isinstance(v, str):
                print(k, v)
            else:
                print(k, v.shape)
        break

    shape_id = '57f73714cbc425e44ae022a8f6e258a7'
    x_rot = R.from_rotvec([-90, 0, 0], degrees=True).as_matrix()
    mesh = o3d.io.read_triangle_mesh('/home/donglin/Data/acronym/meshes/Mug/' + f'{shape_id}.obj')
    mesh.rotate(x_rot)
    mesh.scale(0.018923936541415925, center=[0,0,0])
    
    points = np.asarray(mesh.sample_points_uniformly(20000).points)

    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                points)
    

    mlab.figure(bgcolor=(1, 1, 1))
    draw_scene(
        points,
        grasps=generated_grasps,
        grasp_scores=generated_scores
    )
    mlab.show()