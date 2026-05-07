"""
UE-Pose v2: Uncertainty-driven Event-RGB Fusion for Spacecraft Pose Estimation.

Improved version with:
  1. EPnP everywhere: RC-EDL now uses EPnP for single-channel poses (not BPnP)
  2. Temperature scaling: post-hoc uncertainty calibration parameter
  3. Adaptive weight decay: zero out fusion weights for unreliable channels

Pipeline:
  1. Keypoint-Level Fusion: Gaussian or ACF-OG (with adaptive weight decay)
  2. TAKC: Topology-Aware Keypoint Consistency (optional, --takc)
  3. PnP Pose Solving:
     - Without --ug_ransac: Standard RANSAC + EPnP -> fused pose is final output
     - With --ug_ransac:    UG-RANSAC (weighted LM refinement) + RC-EDL pose selection
       RC-EDL selects among: fused RANSAC, EPnP-RGB, EPnP-Event

Fusion modes (--mode):
  1. gaussian:  Per-keypoint Gaussian-weighted fusion (baseline)
  2. acf_og:    Adaptive Confidence Fusion with Outlier Gating
"""

import json
import numpy as np
import os
import cv2
import torch
import math
import copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import time

import argparse

from kornia.geometry.conversions import rotation_matrix_to_quaternion


def parse_args():
    parser = argparse.ArgumentParser(
        description='UE-Pose v2: Uncertainty-driven Event-RGB Fusion')
    parser.add_argument('--scene',
                        help='scene name',
                        required=True,
                        type=str)

    parser.add_argument('--data_dir',
                        help='base data dir',
                        required=True,
                        type=str)

    parser.add_argument('--mode',
                        help='fusion mode: "gaussian" or "acf_og" (default: acf_og)',
                        default='acf_og',
                        choices=['gaussian', 'acf_og'],
                        type=str)

    parser.add_argument('--alpha',
                        help='[acf_og] outlier gating sensitivity (default: 3.0).',
                        default=3.0,
                        type=float)

    parser.add_argument('--tau_min',
                        help='[acf_og] minimum gating threshold in pixels (default: 5.0).',
                        default=5.0,
                        type=float)

    parser.add_argument('--kp_source',
                        help='keypoint source: "heatmap", "gkr", or "ensemble". Default: heatmap.',
                        default='heatmap',
                        choices=['heatmap', 'gkr', 'ensemble'],
                        type=str)

    # ---- TAKC ----
    parser.add_argument('--takc',
                        help='Enable TAKC post-fusion topology consistency check.',
                        action='store_true',
                        default=False)

    parser.add_argument('--takc_k',
                        help='[TAKC] Number of nearest 3D neighbors per keypoint (default: 5).',
                        default=5,
                        type=int)

    parser.add_argument('--takc_penalty',
                        help='[TAKC] Uncertainty inflation factor (default: 10.0).',
                        default=10.0,
                        type=float)

    parser.add_argument('--takc_thresh',
                        help='[TAKC] Outlier sensitivity as IQR multiplier (default: 2.0).',
                        default=2.0,
                        type=float)

    parser.add_argument('--gate_fallback_thresh',
                        help='[ACF-OG] Frame-level fallback threshold (default: 0.5).',
                        default=0.5,
                        type=float)

    parser.add_argument('--ug_ransac',
                        help='Enable UG-RANSAC + RC-EDL (RANSAC + weighted LM + pose selection).',
                        action='store_true',
                        default=False)

    # ---- RC-EDL parameters ----
    parser.add_argument('--tau_reproj',
                        help='[RC-EDL] Reprojection error scale parameter in pixels (default: 10.0). '
                             'Controls how strongly reprojection error corrects EDL.',
                        default=10.0,
                        type=float)

    parser.add_argument('--cmkd_agree_thresh',
                        help='[RC-EDL] CMKD threshold for channel agreement in pixels (default: 10.0). '
                             'Below this, fused RANSAC pose is considered as a candidate.',
                        default=10.0,
                        type=float)

    # ---- Temperature scaling ----
    parser.add_argument('--temperature',
                        help='Temperature scaling for EDL uncertainty (default: 1.0). '
                             'T>1: less confident (higher uncertainty). '
                             'T<1: more confident (lower uncertainty). '
                             'Applied after cross-channel calibration.',
                        default=1.0,
                        type=float)

    # ---- Adaptive weight decay ----
    parser.add_argument('--weight_decay_thresh',
                        help='Uncertainty threshold for adaptive weight decay (default: 5.0). '
                             'If a channel uncertainty > thresh * median, its fusion weight is zeroed.',
                        default=5.0,
                        type=float)

    args = parser.parse_args()
    return args


# ============================================================================
# Fusion functions (with adaptive weight decay)
# ============================================================================

def gaussian_fuse_adaptive(kp_rgb, kp_event, sigma2_rgb, sigma2_event,
                           weight_decay_thresh=5.0, eps=1e-8):
    """
    Per-keypoint Gaussian-optimal weighted fusion with adaptive weight decay.

    If a channel's uncertainty for keypoint j exceeds weight_decay_thresh times
    the other channel's uncertainty, its weight is zeroed (full gate to the
    more confident channel).

    Args:
        kp_rgb:               (J, 2)
        kp_event:             (J, 2)
        sigma2_rgb:           (J,)  calibrated uncertainty
        sigma2_event:         (J,)  calibrated uncertainty
        weight_decay_thresh:  threshold multiplier for zeroing weights

    Returns:
        kp_fused:      (J, 2)
        sigma2_fused:  (J,)  fused uncertainty
    """
    J = kp_rgb.shape[0]

    # Standard Gaussian weights
    w_rgb = sigma2_event / (sigma2_rgb + sigma2_event + eps)  # (J,)

    # Adaptive weight decay: zero out unreliable channels
    for j in range(J):
        if sigma2_rgb[j] > weight_decay_thresh * sigma2_event[j]:
            # RGB is much more uncertain -> trust event only
            w_rgb[j] = 0.0
        elif sigma2_event[j] > weight_decay_thresh * sigma2_rgb[j]:
            # Event is much more uncertain -> trust RGB only
            w_rgb[j] = 1.0

    w_rgb_2d = w_rgb[:, np.newaxis]  # (J, 1)

    kp_fused = w_rgb_2d * kp_rgb + (1.0 - w_rgb_2d) * kp_event  # (J, 2)

    # Fused uncertainty: for gated keypoints, use the selected channel's uncertainty
    sigma2_fused = np.zeros(J)
    for j in range(J):
        if w_rgb[j] >= 1.0 - eps:
            sigma2_fused[j] = sigma2_rgb[j]
        elif w_rgb[j] <= eps:
            sigma2_fused[j] = sigma2_event[j]
        else:
            sigma2_fused[j] = (sigma2_rgb[j] * sigma2_event[j]) / \
                              (sigma2_rgb[j] + sigma2_event[j] + eps)

    return kp_fused, sigma2_fused


def acf_og_fuse_adaptive(kp_rgb, kp_event, sigma2_rgb, sigma2_event,
                         alpha=3.0, tau_min=5.0, weight_decay_thresh=5.0, eps=1e-8):
    """
    Adaptive Confidence Fusion with Outlier Gating (ACF-OG) + adaptive weight decay.

    For each keypoint j:
      - If sigma2 exceeds weight_decay_thresh * other: zero weight (adaptive decay)
      - d_j > tau_j: outlier -> gate to more confident channel
      - d_j <= tau_j: Gaussian-weighted fusion

    Returns:
        kp_fused, sigma2_fused, gate_mask, gate_source
    """
    J = kp_rgb.shape[0]

    cmkd = np.linalg.norm(kp_rgb - kp_event, axis=-1)  # (J,)
    sigma_rgb = np.sqrt(sigma2_rgb + eps)
    sigma_event = np.sqrt(sigma2_event + eps)
    tau = np.maximum(alpha * (sigma_rgb + sigma_event), tau_min)

    gate_mask = cmkd > tau
    w_rgb = sigma2_event / (sigma2_rgb + sigma2_event + eps)

    kp_fused = np.zeros_like(kp_rgb)
    sigma2_fused = np.zeros(J)
    gate_source = np.zeros(J, dtype=int)

    for j in range(J):
        # Adaptive weight decay: zero out unreliable channels regardless of CMKD
        if sigma2_rgb[j] > weight_decay_thresh * sigma2_event[j]:
            # RGB is much more uncertain -> always trust event
            kp_fused[j] = kp_event[j]
            sigma2_fused[j] = sigma2_event[j]
            gate_source[j] = 2
            gate_mask[j] = True
        elif sigma2_event[j] > weight_decay_thresh * sigma2_rgb[j]:
            # Event is much more uncertain -> always trust RGB
            kp_fused[j] = kp_rgb[j]
            sigma2_fused[j] = sigma2_rgb[j]
            gate_source[j] = 1
            gate_mask[j] = True
        elif gate_mask[j]:
            # Standard outlier gating
            if sigma2_rgb[j] < sigma2_event[j]:
                kp_fused[j] = kp_rgb[j]
                sigma2_fused[j] = sigma2_rgb[j]
                gate_source[j] = 1
            else:
                kp_fused[j] = kp_event[j]
                sigma2_fused[j] = sigma2_event[j]
                gate_source[j] = 2
        else:
            # Standard Gaussian fusion
            w = w_rgb[j]
            kp_fused[j] = w * kp_rgb[j] + (1.0 - w) * kp_event[j]
            sigma2_fused[j] = (sigma2_rgb[j] * sigma2_event[j]) / \
                              (sigma2_rgb[j] + sigma2_event[j] + eps)
            gate_source[j] = 0

    return kp_fused, sigma2_fused, gate_mask, gate_source


# ============================================================================
# UG-RANSAC
# ============================================================================

def ug_ransac_pnp(pts_3d, pts_2d, K, sigma2, dist_coeffs,
                  reproj_thresh=20.0, ransac_iters=10000):
    """
    Uncertainty-Guided RANSAC for PnP.
    Step 1: Standard RANSAC -> robust initial pose
    Step 2: Uncertainty-weighted LM refinement (inliers only)
    """
    J = pts_3d.shape[0]

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=ransac_iters,
        reprojectionError=reproj_thresh)

    if not ret:
        return ret, rvec, tvec, inliers

    # Only refine using RANSAC inliers
    if inliers is None or len(inliers) < 4:
        return ret, rvec, tvec, inliers

    inlier_idx = inliers.flatten()
    pts_3d_inlier = pts_3d[inlier_idx].astype(np.float64)
    pts_2d_inlier = pts_2d[inlier_idx].astype(np.float64)
    sigma2_inlier = sigma2[inlier_idx]

    eps = 1e-8
    weights = np.sqrt(1.0 / (sigma2_inlier + eps))
    weights = weights / (weights.mean() + eps)

    x0 = np.concatenate([rvec.flatten(), tvec.flatten()])

    def weighted_reproj_residuals(x):
        rv = x[:3].reshape(3, 1)
        tv = x[3:6].reshape(3, 1)
        projected, _ = cv2.projectPoints(
            pts_3d_inlier, rv, tv, K, dist_coeffs)
        projected = projected.reshape(-1, 2)
        residuals = (pts_2d_inlier - projected)
        residuals = residuals * weights[:, np.newaxis]
        return residuals.flatten()

    try:
        from scipy.optimize import least_squares
        result = least_squares(weighted_reproj_residuals, x0, method='lm')
        if result.success:
            rvec = result.x[:3].reshape(3, 1)
            tvec = result.x[3:6].reshape(3, 1)
    except Exception:
        pass

    return ret, rvec, tvec, inliers


# ============================================================================
# EPnP Pose Solver (replaces BPnP in RC-EDL)
# ============================================================================

def epnp_solve(kp_2d, landmarks_3d, K, dist_coeffs,
               ransac_iters=10000, reproj_thresh=20.0):
    """
    Solve pose using RANSAC + EPnP for a single image.

    Args:
        kp_2d:          (J, 2) 2D keypoint coordinates
        landmarks_3d:   (J, 3) 3D landmark positions
        K:              (3, 3) camera intrinsic matrix
        dist_coeffs:    distortion coefficients
        ransac_iters:   RANSAC iterations
        reproj_thresh:  reprojection error threshold

    Returns:
        pose_4x4:  (4, 4) pose matrix, identity if failed
        success:   bool
    """
    pts_3d = landmarks_3d.astype(np.float64)
    pts_2d = kp_2d.astype(np.float64).reshape(-1, 1, 2)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K.astype(np.float64), dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=ransac_iters,
        reprojectionError=reproj_thresh)

    pose = np.eye(4, dtype=np.float64)
    if ret:
        R, _ = cv2.Rodrigues(rvec)
        pose[0:3, 0:3] = R
        pose[0:3, 3] = tvec.flatten()

    return pose, ret


# ============================================================================
# TAKC: Topology-Aware Keypoint Consistency
# ============================================================================

def build_topology_prior(landmarks_3d, k=5):
    """Build topology prior from 3D landmark positions."""
    J = landmarks_3d.shape[0]
    dist_3d = cdist(landmarks_3d, landmarks_3d)
    k = min(k, J - 1)
    neighbors = np.zeros((J, k), dtype=int)
    for j in range(J):
        sorted_idx = np.argsort(dist_3d[j])
        neighbors[j] = sorted_idx[1:k+1]
    return neighbors, dist_3d


def takc_consistency_score(kp_2d, neighbors, dist_3d, thresh=0.3):
    """Compute per-keypoint topology consistency score."""
    J, K = neighbors.shape
    scores = np.zeros(J)
    dist_2d = cdist(kp_2d, kp_2d)

    for j in range(J):
        nbrs = neighbors[j]
        d3 = dist_3d[j, nbrs]
        d2 = dist_2d[j, nbrs]

        if d3.sum() < 1e-8 or d2.sum() < 1e-8:
            scores[j] = 0.0
            continue

        r3 = d3 / (d3.sum() + 1e-8)
        r2 = d2 / (d2.sum() + 1e-8)
        deviation = np.mean(np.abs(r2 - r3))
        scores[j] = deviation

    q25 = np.percentile(scores, 25)
    q75 = np.percentile(scores, 75)
    iqr = q75 - q25 + 1e-8
    adaptive_thresh = np.median(scores) + thresh * iqr
    outlier_mask = scores > adaptive_thresh

    return scores, outlier_mask


# ============================================================================
# RC-EDL: Reprojection-Calibrated EDL Decision
# ============================================================================

def compute_self_reproj_error(pose_4x4, kp_2d, landmarks_3d, K, dist_coeffs):
    """
    Compute median reprojection error of a pose against its OWN keypoints.
    This measures self-consistency: a good pose should reproject close to
    the keypoints that generated it.
    """
    rvec, _ = cv2.Rodrigues(pose_4x4[:3, :3].astype(np.float64))
    tvec = pose_4x4[:3, 3].reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(
        landmarks_3d.astype(np.float64), rvec, tvec,
        K.astype(np.float64), dist_coeffs)
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(proj - kp_2d, axis=1)
    return np.median(errors)


def rc_edl_score(edl_uncertainty_mean, reproj_error, tau_reproj=10.0, eps=1e-8):
    """
    Compute RC-EDL reliability score for a single channel.

    combined_score = edl_score * reproj_factor

    Args:
        edl_uncertainty_mean: mean calibrated sigma2 for this channel (lower = more confident)
        reproj_error:         median reprojection error in pixels (lower = more consistent)
        tau_reproj:           scale parameter for reprojection penalty

    Returns:
        combined_score: higher = more reliable
    """
    edl_score = 1.0 / (edl_uncertainty_mean + eps)
    reproj_factor = 1.0 / (1.0 + reproj_error / tau_reproj)
    return edl_score * reproj_factor


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    scene = args.scene
    data_dir = args.data_dir
    mode = args.mode
    alpha = args.alpha
    tau_min = args.tau_min
    kp_source = args.kp_source
    use_takc = args.takc
    takc_k = args.takc_k
    takc_penalty = args.takc_penalty
    takc_thresh = args.takc_thresh
    use_ug_ransac = args.ug_ransac
    gate_fallback_thresh = args.gate_fallback_thresh
    tau_reproj = args.tau_reproj
    cmkd_agree_thresh = args.cmkd_agree_thresh
    temperature = args.temperature
    weight_decay_thresh = args.weight_decay_thresh

    print(f'Scene: {scene}')
    print(f'Fusion mode: {mode}, Keypoint source: {kp_source}')
    print(f'Temperature scaling: T={temperature}')
    print(f'Adaptive weight decay threshold: {weight_decay_thresh}x')
    if mode == 'acf_og':
        print(f'  ACF-OG params: alpha={alpha}, tau_min={tau_min}')
        print(f'  Frame-level fallback threshold: {gate_fallback_thresh}')
    if use_takc:
        print(f'  TAKC enabled: K={takc_k}, thresh={takc_thresh}, penalty={takc_penalty}')
    if use_ug_ransac:
        print(f'  UG-RANSAC + RC-EDL enabled')
        print(f'    UG-RANSAC: RANSAC + weighted LM refinement (w=1/sigma2)')
        print(f'    RC-EDL: tau_reproj={tau_reproj}, cmkd_agree={cmkd_agree_thresh}')
        print(f'    Single-channel poses: EPnP (replaces BPnP)')
    else:
        print(f'  Standard RANSAC + EPnP (no UG-RANSAC, no RC-EDL)')
        print(f'  Standard RANSAC: iters=10000, reproj=20')

    satellite = scene.split('-')[0]+ "_cal"
    ransaciters = 10000
    reprojection_error_thresh = 20

    # ---- EDL predictions file ----
    predictions_file_edl = 'new-results/{}/EventRGBAlignedDataset/pose_efficientnet_edl/{}/predictions.json'.format(
        satellite, scene)

    with open(predictions_file_edl, 'r') as jsonfile:
        predictions_edl = json.load(jsonfile)

    predictions_fusion = copy.deepcopy(predictions_edl)

    # GT file
    with open(os.path.join('../../../data/new-data/', data_dir, scene, 'test.json'), 'r') as jsonfile:
        gt_labels_all = json.load(jsonfile)

    # ---- Read per-keypoint EDL uncertainties ----
    num_images = len(predictions_edl['annotations'])
    num_joints = len(predictions_edl['annotations'][0]['keypoint_variances_rgb'])

    uncertainties_rgb = np.zeros((num_images, num_joints))
    uncertainties_event = np.zeros((num_images, num_joints))

    for image_index in range(num_images):
        ann = predictions_edl['annotations'][image_index]
        var_rgb = np.array(ann['keypoint_variances_rgb'])
        var_event = np.array(ann['keypoint_variances_event'])
        uncertainties_rgb[image_index, :] = np.mean(var_rgb, axis=-1)
        uncertainties_event[image_index, :] = np.mean(var_event, axis=-1)

    # ---- Cross-channel uncertainty calibration ----
    median_rgb = np.median(uncertainties_rgb) + 1e-8
    median_event = np.median(uncertainties_event) + 1e-8
    print(f'Raw median sigma2  -- RGB: {median_rgb:.6f}, Event: {median_event:.6f}, '
          f'ratio: {median_event/median_rgb:.3f}')

    uncertainties_rgb_cal = uncertainties_rgb / median_rgb
    uncertainties_event_cal = uncertainties_event / median_event
    print(f'After calibration: both channels median -> 1.0')

    # ---- Apply temperature scaling ----
    if temperature != 1.0:
        uncertainties_rgb_cal *= temperature
        uncertainties_event_cal *= temperature
        print(f'After temperature scaling (T={temperature}): '
              f'median RGB={np.median(uncertainties_rgb_cal):.4f}, '
              f'median Event={np.median(uncertainties_event_cal):.4f}')

    # ---- Read keypoint predictions ----
    keypoints_rgb = np.array([
        predictions_edl['annotations'][i]['keypoints_rgb']
        for i in range(num_images)
    ])
    keypoints_event = np.array([
        predictions_edl['annotations'][i]['keypoints_event']
        for i in range(num_images)
    ])
    if keypoints_rgb.ndim == 4:
        keypoints_rgb = keypoints_rgb.squeeze(axis=2)
    if keypoints_event.ndim == 4:
        keypoints_event = keypoints_event.squeeze(axis=2)

    # ---- GKR keypoints ----
    has_gkr = 'gkr_keypoints_rgb' in predictions_edl['annotations'][0]
    if kp_source in ('gkr', 'ensemble') and not has_gkr:
        print('WARNING: --kp_source={} requested but no gkr_keypoints. Falling back to heatmap.'.format(kp_source))
        kp_source = 'heatmap'

    if has_gkr:
        gkr_kp_rgb = np.array([
            predictions_edl['annotations'][i]['gkr_keypoints_rgb']
            for i in range(num_images)
        ])
        gkr_kp_event = np.array([
            predictions_edl['annotations'][i]['gkr_keypoints_event']
            for i in range(num_images)
        ])
        print(f'GKR keypoints loaded: {gkr_kp_rgb.shape}')

    if kp_source == 'gkr':
        keypoints_rgb = gkr_kp_rgb
        keypoints_event = gkr_kp_event
        print('Using GKR refined keypoints')
    elif kp_source == 'ensemble':
        keypoints_rgb = 0.5 * keypoints_rgb + 0.5 * gkr_kp_rgb
        keypoints_event = 0.5 * keypoints_event + 0.5 * gkr_kp_event
        print('Using ensemble keypoints')
    else:
        print('Using heatmap argmax keypoints')

    instance_count = len(gt_labels_all['annotations'])
    print(f'Total instance count: {instance_count}')

    landmarks_3d = np.array(gt_labels_all['landmarks_3d'])
    distortion_coefficients = np.array([0., 0., 0., 0., 0.])
    K_rgb = np.array(gt_labels_all['intrinsics'])

    # ---- Build TAKC topology prior ----
    if use_takc:
        takc_neighbors, takc_dist_3d = build_topology_prior(landmarks_3d, k=takc_k)
        print(f'TAKC topology prior built: {landmarks_3d.shape[0]} keypoints, '
              f'K={takc_k} neighbors each')

    # ---- Statistics counters ----
    avg_time = 0.0
    total_gated_keypoints = 0
    total_fused_keypoints = 0
    total_gated_to_rgb = 0
    total_gated_to_event = 0
    per_image_gate_ratio = []
    total_takc_outliers = 0
    total_takc_checked = 0
    total_frame_fallbacks = 0
    total_pose_fused = 0
    total_pose_epnp_rgb = 0
    total_pose_epnp_event = 0
    total_weight_decay_applied = 0

    for image_index_single in tqdm(range(instance_count)):
        start_time = time.time()

        kp_rgb = keypoints_rgb[image_index_single, :, :]      # (J, 2)
        kp_event = keypoints_event[image_index_single, :, :]  # (J, 2)
        sigma2_rgb = uncertainties_rgb_cal[image_index_single, :]   # (J,)
        sigma2_event = uncertainties_event_cal[image_index_single, :]  # (J,)

        # Track weight decay usage
        n_decayed = np.sum(
            (sigma2_rgb > weight_decay_thresh * sigma2_event) |
            (sigma2_event > weight_decay_thresh * sigma2_rgb)
        )
        total_weight_decay_applied += n_decayed

        # ================================================================
        # Stage 1: Keypoint-Level Fusion (with adaptive weight decay)
        # ================================================================
        if mode == 'gaussian':
            kp_fused, sigma2_fused = gaussian_fuse_adaptive(
                kp_rgb, kp_event, sigma2_rgb, sigma2_event,
                weight_decay_thresh=weight_decay_thresh)
            gate_source = np.zeros(kp_rgb.shape[0], dtype=int)
        elif mode == 'acf_og':
            kp_fused, sigma2_fused, gate_mask, gate_source = acf_og_fuse_adaptive(
                kp_rgb, kp_event, sigma2_rgb, sigma2_event,
                alpha=alpha, tau_min=tau_min,
                weight_decay_thresh=weight_decay_thresh)
            n_gated = gate_mask.sum()
            total_gated_keypoints += n_gated
            total_fused_keypoints += (~gate_mask).sum()
            total_gated_to_rgb += (gate_source == 1).sum()
            total_gated_to_event += (gate_source == 2).sum()
            per_image_gate_ratio.append(n_gated / len(gate_mask))

            # Frame-level fallback for severe disagreement
            gate_ratio = n_gated / len(gate_mask)
            if gate_ratio > gate_fallback_thresh:
                if sigma2_rgb.mean() < sigma2_event.mean():
                    kp_fused = kp_rgb.copy()
                    sigma2_fused = sigma2_rgb.copy()
                    gate_source = np.ones(len(gate_mask), dtype=int)
                else:
                    kp_fused = kp_event.copy()
                    sigma2_fused = sigma2_event.copy()
                    gate_source = np.full(len(gate_mask), 2, dtype=int)
                total_frame_fallbacks += 1

        # ================================================================
        # Stage 2: TAKC post-processing
        # ================================================================
        if use_takc:
            scores, outlier_mask = takc_consistency_score(
                kp_fused, takc_neighbors, takc_dist_3d, thresh=takc_thresh)
            sigma2_fused[outlier_mask] *= takc_penalty
            total_takc_outliers += outlier_mask.sum()
            total_takc_checked += len(outlier_mask)

            consistent_mask = ~outlier_mask
            if consistent_mask.sum() >= 4:
                kp_pnp = kp_fused[consistent_mask]
                lm_pnp = landmarks_3d[consistent_mask]
            else:
                kp_pnp = kp_fused
                lm_pnp = landmarks_3d
        else:
            kp_pnp = kp_fused
            lm_pnp = landmarks_3d

        # ================================================================
        # Stage 3: PnP (UG-RANSAC or standard)
        # ================================================================
        if use_ug_ransac:
            if use_takc and consistent_mask.sum() >= 4:
                sigma2_pnp = sigma2_fused[consistent_mask]
            else:
                sigma2_pnp = sigma2_fused

            ret, pred_rotation_vector, pred_translation, inliers = ug_ransac_pnp(
                lm_pnp, kp_pnp, K_rgb, sigma2_pnp,
                dist_coeffs=distortion_coefficients,
                reproj_thresh=reprojection_error_thresh,
                ransac_iters=ransaciters)
        else:
            ret, pred_rotation_vector, pred_translation, inliers = cv2.solvePnPRansac(
                lm_pnp, kp_pnp, K_rgb,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=ransaciters,
                reprojectionError=reprojection_error_thresh,
                distCoeffs=distortion_coefficients)

        time_taken = time.time() - start_time
        avg_time += time_taken

        pred_rotation_matrix, _ = cv2.Rodrigues(pred_rotation_vector)
        rt = np.column_stack((pred_rotation_matrix, pred_translation))
        ransac_pose = np.eye(4)
        ransac_pose[0:3, :] = rt

        # ================================================================
        # Stage 4: Pose Selection
        # ================================================================
        mean_cmkd = np.linalg.norm(kp_rgb - kp_event, axis=-1).mean()
        n_inliers = len(inliers) if inliers is not None else 0
        inlier_ratio = n_inliers / kp_pnp.shape[0] if kp_pnp.shape[0] > 0 else 0

        if use_ug_ransac:
            # ---- RC-EDL Pose Selection (with UG-RANSAC) ----
            # 3 candidate poses — ALL using EPnP (not BPnP):
            #   1. epnp_pose_rgb   (EPnP on single-channel RGB keypoints)
            #   2. epnp_pose_event (EPnP on single-channel event keypoints)
            #   3. ransac_pose     (from fused keypoints + UG-RANSAC)

            # Compute EPnP poses for single channels (replaces BPnP)
            epnp_pose_rgb, ret_rgb = epnp_solve(
                kp_rgb, landmarks_3d, K_rgb, distortion_coefficients,
                ransac_iters=ransaciters, reproj_thresh=reprojection_error_thresh)
            epnp_pose_event, ret_event = epnp_solve(
                kp_event, landmarks_3d, K_rgb, distortion_coefficients,
                ransac_iters=ransaciters, reproj_thresh=reprojection_error_thresh)

            # EDL scores (primary signal)
            edl_mean_rgb = sigma2_rgb.mean()
            edl_mean_event = sigma2_event.mean()

            # Self-consistency reprojection errors
            reproj_rgb = compute_self_reproj_error(
                epnp_pose_rgb, kp_rgb, landmarks_3d, K_rgb, distortion_coefficients)
            reproj_event = compute_self_reproj_error(
                epnp_pose_event, kp_event, landmarks_3d, K_rgb, distortion_coefficients)

            # RC-EDL combined scores
            score_rgb = rc_edl_score(edl_mean_rgb, reproj_rgb, tau_reproj)
            score_event = rc_edl_score(edl_mean_event, reproj_event, tau_reproj)

            # Fused pose: check against FUSED keypoints (fusion-method-dependent)
            reproj_fused = compute_self_reproj_error(
                ransac_pose, kp_fused, landmarks_3d, K_rgb, distortion_coefficients)
            edl_mean_fused = (edl_mean_rgb + edl_mean_event) / 2.0
            score_fused = rc_edl_score(edl_mean_fused, reproj_fused, tau_reproj)

            # Pick the candidate with the highest RC-EDL score
            best_score = max(score_rgb, score_event, score_fused)

            if score_fused == best_score:
                final_pose = ransac_pose
                pose_source = 'fused'
                total_pose_fused += 1
            elif score_rgb >= score_event:
                final_pose = epnp_pose_rgb
                pose_source = 'epnp_rgb'
                total_pose_epnp_rgb += 1
            else:
                final_pose = epnp_pose_event
                pose_source = 'epnp_event'
                total_pose_epnp_event += 1

            # Store RC-EDL specific results
            predictions_fusion['annotations'][image_index_single]['pose_source'] = pose_source
            predictions_fusion['annotations'][image_index_single]['rc_edl_score_rgb'] = float(score_rgb)
            predictions_fusion['annotations'][image_index_single]['rc_edl_score_event'] = float(score_event)
            predictions_fusion['annotations'][image_index_single]['rc_edl_score_fused'] = float(score_fused)
            # Store EPnP single-channel poses for reference
            predictions_fusion['annotations'][image_index_single]['pose_epnp_rgb'] = epnp_pose_rgb.tolist()
            predictions_fusion['annotations'][image_index_single]['pose_epnp_event'] = epnp_pose_event.tolist()
        else:
            # ---- No RC-EDL: fused RANSAC pose is the final output ----
            final_pose = ransac_pose
            pose_source = 'fused'
            total_pose_fused += 1

        # ---- Store results ----
        predictions_fusion['annotations'][image_index_single]['pose_fused'] = final_pose.tolist()
        # Preserve original BPnP poses (don't overwrite with final_pose)
        # pose_rgb and pose_event remain as the single-channel BPnP results from test stage

        # Store fused keypoints in the same format as keypoints_rgb/event: [[[x, y]], ...]
        kp_fused_stored = [[[float(kp_fused[j, 0]), float(kp_fused[j, 1])]]
                           for j in range(kp_fused.shape[0])]
        predictions_fusion['annotations'][image_index_single]['keypoints_fused'] = kp_fused_stored

        predictions_fusion['annotations'][image_index_single]['keypoint_variances_fused'] = \
            sigma2_fused.tolist()
        predictions_fusion['annotations'][image_index_single]['fusion_gate_source'] = \
            gate_source.tolist()
        predictions_fusion['annotations'][image_index_single]['mean_cmkd'] = float(mean_cmkd)
        predictions_fusion['annotations'][image_index_single]['ransac_inlier_ratio'] = float(inlier_ratio)

        if use_takc:
            predictions_fusion['annotations'][image_index_single]['takc_scores'] = \
                scores.tolist()
            predictions_fusion['annotations'][image_index_single]['takc_outliers'] = \
                outlier_mask.tolist()

    avg_time /= instance_count

    # ---- Print statistics ----
    print(f'\n{"="*60}')
    if mode == 'gaussian':
        print(f'Gaussian Weighted Fusion (with calibration + adaptive decay)')
        print(f'{"="*60}')
        print(f'  All {instance_count * keypoints_rgb.shape[1]} keypoints Gaussian-fused')
    elif mode == 'acf_og':
        total_kp = total_gated_keypoints + total_fused_keypoints
        print(f'ACF-OG Fusion Statistics (with adaptive weight decay)')
        print(f'{"="*60}')
        print(f'  alpha = {alpha}, tau_min = {tau_min} px')
        print(f'  Total keypoints processed:   {total_kp}')
        print(f'  Gaussian-fused (agreement):  {total_fused_keypoints} '
              f'({100*total_fused_keypoints/total_kp:.1f}%)')
        print(f'  Outlier-gated (disagreement):{total_gated_keypoints} '
              f'({100*total_gated_keypoints/total_kp:.1f}%)')
        print(f'    -> gated to RGB:   {total_gated_to_rgb}')
        print(f'    -> gated to Event: {total_gated_to_event}')
        print(f'  Per-image gate ratio: mean={np.mean(per_image_gate_ratio):.3f}, '
              f'max={np.max(per_image_gate_ratio):.3f}')
        print(f'  Frame-level fallbacks: {total_frame_fallbacks}/{instance_count} '
              f'({100*total_frame_fallbacks/instance_count:.1f}%) '
              f'[thresh={gate_fallback_thresh}]')

    total_kp_all = instance_count * keypoints_rgb.shape[1]
    print(f'  Adaptive weight decay: {total_weight_decay_applied}/{total_kp_all} keypoints '
          f'({100*total_weight_decay_applied/total_kp_all:.1f}%) '
          f'[thresh={weight_decay_thresh}x]')
    print(f'  Temperature scaling: T={temperature}')

    if use_ug_ransac:
        print(f'  RC-EDL Pose selection (EPnP for single-channel):')
        print(f'    Fused RANSAC pose:  {total_pose_fused}/{instance_count} '
              f'({100*total_pose_fused/instance_count:.1f}%)')
        print(f'    EPnP RGB fallback:  {total_pose_epnp_rgb}/{instance_count} '
              f'({100*total_pose_epnp_rgb/instance_count:.1f}%)')
        print(f'    EPnP Event fallback:{total_pose_epnp_event}/{instance_count} '
              f'({100*total_pose_epnp_event/instance_count:.1f}%)')
        print(f'  UG-RANSAC: RANSAC + weighted LM refinement (w=1/sigma2)')
    else:
        print(f'  Pose: fused RANSAC pose used directly (no RC-EDL selection)')
        print(f'  Standard RANSAC: iters={ransaciters}, reproj={reprojection_error_thresh}')

    if use_takc:
        print(f'  TAKC: {total_takc_outliers}/{total_takc_checked} keypoints flagged '
              f'({100*total_takc_outliers/max(total_takc_checked,1):.1f}%)'
              f'  [K={takc_k}, thresh={takc_thresh}, penalty={takc_penalty}]')
    print(f'  Avg time per image: {avg_time:.6f}s')
    print(f'{"="*60}')

    # ---- Save results ----
    output_dir = os.path.dirname(predictions_file_edl)
    kp_tag = '' if kp_source == 'heatmap' else f'_{kp_source}'
    takc_tag = '_takc' if use_takc else ''
    ug_tag = '_ugransac' if use_ug_ransac else ''
    temp_tag = f'_T{temperature}' if temperature != 1.0 else ''
    if mode == 'gaussian':
        output_filename = f'predictions_gaussian{kp_tag}{takc_tag}{ug_tag}{temp_tag}_fusion.json'
    else:
        output_filename = f'predictions_acf_og{kp_tag}{takc_tag}{ug_tag}{temp_tag}_fusion.json'
    final_output_path = os.path.join(output_dir, output_filename)

    with open(final_output_path, 'w') as jsonfile:
        jsonfile.write(json.dumps(predictions_fusion, indent=2))

    print(f'Saved {mode} fusion results to: {final_output_path}')


if __name__ == '__main__':
    main()
