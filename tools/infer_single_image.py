# ------------------------------------------------------------------------------
# Single-Image Keypoint Detection & Visualization
# Loads a trained EDL checkpoint and runs inference on one image.
# No dataset JSON, no ground truth needed — pure visual inspection.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.inference import get_final_preds
from core.edl_loss import NIGLoss
from utils.transforms import (
    get_affine_transform,
    FlipBlackEventsToWhite,
    FillEventBlack,
)

import models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single-image keypoint detection with EDL uncertainty'
    )
    parser.add_argument(
        '--cfg', required=True, type=str,
        help='experiment YAML config file (e.g. experiments/generic/efficientnet/train.yaml)'
    )
    parser.add_argument(
        '--checkpoint', required=True, type=str,
        help='path to checkpoint file (checkpoint_event.pth or checkpoint_rgb.pth)'
    )
    parser.add_argument(
        '--image', required=True, type=str,
        help='path to input image (event frame or RGB image)'
    )
    parser.add_argument(
        '--branch', default='rgb', choices=['event', 'rgb'],
        help='which branch to use: event or rgb (determines preprocessing)'
    )
    parser.add_argument(
        '--bbox', default=None, type=str,
        help='bounding box as x1,y1,x2,y2 (default: full image)'
    )
    parser.add_argument(
        '--output', default='output_keypoints.png', type=str,
        help='output visualization path'
    )
    parser.add_argument('--modelDir', default='', type=str,
                        help='model directory')
    parser.add_argument('--logDir', default='', type=str,
                        help='log directory')
    parser.add_argument('--dataDir', default='', type=str,
                        help='data directory')
    parser.add_argument('--prevModelDir', default='', type=str,
                        help='prev model directory')
    parser.add_argument(
        'opts', nargs=argparse.REMAINDER, default=None,
        help='modify config options from command line'
    )
    return parser.parse_args()


def box2cs(box, aspect_ratio=1.0, pixel_std=200):
    """Convert bounding box [x1, y1, x2, y2] to center + scale."""
    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    # make square
    if w > h:
        h = w
    else:
        w = h

    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32
    )
    scale = scale * 1.2  # margin
    return center, scale


def build_event_transform():
    """Preprocessing pipeline for event frames (matches test script)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        FlipBlackEventsToWhite(),
        FillEventBlack(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])


def build_rgb_transform():
    """Preprocessing pipeline for RGB frames (matches test script)."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def main():
    args = parse_args()

    # --- Config ---
    update_config(cfg, args)

    input_size = np.array(cfg.MODEL.IMAGE_SIZE)    # e.g. [512, 512]
    heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)  # e.g. [128, 128]

    # --- Build model ---
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False, edl=True
    )
    encoder = eval('models.' + cfg.MODEL.NAME + '.get_encoder')(
        cfg, is_train=False, dropout_prob=0.1
    )

    model = torch.nn.DataParallel(model).cuda()
    encoder = torch.nn.DataParallel(encoder).cuda()

    # --- Load checkpoint ---
    print(f'=> Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if args.branch == 'event':
        state_key = 'state_dict_event'
        encoder_key = 'state_dict_encoder_event'
    else:
        state_key = 'state_dict_rgb'
        encoder_key = 'state_dict_encoder_rgb'

    model.load_state_dict(checkpoint[state_key])
    encoder.load_state_dict(checkpoint[encoder_key])
    epoch = checkpoint.get('epoch', '?')
    print(f'=> Loaded checkpoint (epoch {epoch})')

    model.eval()
    encoder.eval()

    # --- Read image ---
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image_bgr is None:
        print(f'ERROR: cannot read image: {args.image}')
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    # --- Bounding box → center + scale ---
    if args.bbox:
        bbox = [float(v) for v in args.bbox.split(',')]
        assert len(bbox) == 4, 'bbox must be x1,y1,x2,y2'
    else:
        bbox = [0, 0, w_orig, h_orig]

    center, scale = box2cs(bbox)

    # --- Crop + resize (same transform as dataset) ---
    trans = get_affine_transform(center, scale, 0, input_size)
    trans_inv = get_affine_transform(center, scale, 0, heatmap_size, inv=1)

    cropped = cv2.warpAffine(
        image_rgb, trans,
        (int(input_size[0]), int(input_size[1])),
        flags=cv2.INTER_LINEAR,
        borderValue=(127, 127, 127) if args.branch == 'event' else (0, 0, 0)
    )

    # --- Apply transforms ---
    if args.branch == 'event':
        transform = build_event_transform()
    else:
        transform = build_rgb_transform()

    input_tensor = transform(cropped)  # (3, H, W)
    input_tensor = input_tensor.unsqueeze(0).cuda()  # (1, 3, H, W)

    # --- Inference ---
    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = model(features)

        if isinstance(outputs, tuple):
            heatmaps, nig_params = outputs
        else:
            heatmaps = outputs
            nig_params = None

    # --- Extract keypoints ---
    center_batch = center.reshape(1, 2)
    scale_batch = scale.reshape(1, 2)

    preds, maxvals = get_final_preds(
        cfg, heatmaps.cpu().numpy(), center_batch, scale_batch
    )
    # preds: (1, NUM_JOINTS, 2), maxvals: (1, NUM_JOINTS, 1)
    keypoints = preds[0]      # (NUM_JOINTS, 2)
    confidences = maxvals[0]  # (NUM_JOINTS, 1)

    # --- EDL uncertainty ---
    uncertainties = None
    if nig_params is not None:
        epistemic = NIGLoss.compute_uncertainty(nig_params)  # (1, NUM_JOINTS)
        uncertainties = epistemic[0].cpu().numpy()  # (NUM_JOINTS,)

    # --- Print results ---
    num_joints = keypoints.shape[0]
    print(f'\n{"="*60}')
    print(f'Detected {num_joints} keypoints:')
    print(f'{"="*60}')
    print(f'{"Joint":>6}  {"X":>8}  {"Y":>8}  {"Conf":>8}  {"Uncert":>8}')
    print(f'{"-"*6}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}')
    for j in range(num_joints):
        unc_str = f'{uncertainties[j]:.4f}' if uncertainties is not None else 'N/A'
        print(f'{j:>6}  {keypoints[j, 0]:>8.1f}  {keypoints[j, 1]:>8.1f}  '
              f'{confidences[j, 0]:>8.4f}  {unc_str:>8}')
    print(f'{"="*60}\n')

    # --- Save predicted heatmaps ---
    hm_np = heatmaps[0].cpu().numpy()  # (NUM_JOINTS, H_hm, W_hm)

    base, ext = os.path.splitext(args.output)
    hm_dir = base + '_heatmaps'
    os.makedirs(hm_dir, exist_ok=True)

    # 1) Raw numpy (for downstream use)
    npy_path = os.path.join(hm_dir, 'heatmaps_raw.npy')
    np.save(npy_path, hm_np)
    print(f'Raw heatmaps saved to: {npy_path}  (shape {hm_np.shape})')

    # 2) Per-joint colorized heatmap images
    for j in range(num_joints):
        hm_j = hm_np[j]  # (H_hm, W_hm)
        # Normalize to 0-255
        hm_min, hm_max = hm_j.min(), hm_j.max()
        if hm_max - hm_min > 1e-6:
            hm_norm = ((hm_j - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros_like(hm_j, dtype=np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(hm_dir, f'joint_{j:02d}.png'), hm_color)

    # 3) Summary grid of all joints
    hm_h, hm_w = hm_np.shape[1], hm_np.shape[2]
    cols = min(num_joints, 6)
    rows = (num_joints + cols - 1) // cols
    grid = np.zeros((rows * hm_h, cols * hm_w, 3), dtype=np.uint8)
    for j in range(num_joints):
        r, c = j // cols, j % cols
        hm_j = hm_np[j]
        hm_min, hm_max = hm_j.min(), hm_j.max()
        if hm_max - hm_min > 1e-6:
            hm_norm = ((hm_j - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros_like(hm_j, dtype=np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        cv2.putText(hm_color, str(j), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        grid[r * hm_h:(r + 1) * hm_h, c * hm_w:(c + 1) * hm_w] = hm_color
    cv2.imwrite(os.path.join(hm_dir, 'grid_all_joints.png'), grid)

    # 4) Combined heatmap overlay on original image
    #    Heatmap is in cropped/transformed space (128x128).
    #    Must warp it back to original image coords via inverse affine.
    hm_sum = hm_np.max(axis=0)  # (H_hm, W_hm) — max across joints
    hm_min, hm_max = hm_sum.min(), hm_sum.max()
    if hm_max - hm_min > 1e-6:
        hm_norm = ((hm_sum - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
    else:
        hm_norm = np.zeros_like(hm_sum, dtype=np.uint8)

    # Inverse affine: heatmap coords → original image coords
    trans_hm_to_orig = get_affine_transform(center, scale, 0, heatmap_size, inv=1)
    hm_in_orig = cv2.warpAffine(
        hm_norm, trans_hm_to_orig, (w_orig, h_orig),
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )
    hm_color_overlay = cv2.applyColorMap(hm_in_orig, cv2.COLORMAP_JET)
    # Mask out background (where heatmap is 0) to keep original image clear
    mask = (hm_in_orig > 5).astype(np.float32)[..., None]
    overlay_img = (image_bgr.astype(np.float32) * (1 - mask * 0.4) +
                   hm_color_overlay.astype(np.float32) * mask * 0.4).astype(np.uint8)
    cv2.imwrite(os.path.join(hm_dir, 'overlay_on_image.png'), overlay_img)

    print(f'Heatmap visualizations saved to: {hm_dir}/')


    # --- Output 1: Red keypoint dots only ---
    vis_kpts = image_bgr.copy()
    RED = (0, 0, 255)  # BGR

    for j in range(num_joints):
        x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
        conf = confidences[j, 0]
        if conf < 0.01:
            continue
        cv2.circle(vis_kpts, (x, y), 5, RED, -1, cv2.LINE_AA)

    output_kpts = args.output
    cv2.imwrite(output_kpts, vis_kpts)
    print(f'Keypoint visualization saved to: {output_kpts}')

    # --- Output 2: Wireframe (convex hull outline) ---
    vis_wire = image_bgr.copy()
    WIRE_COLOR = (255, 103, 0)  # dark orange in BGR (similar to Figure 3 style)

    # Collect valid keypoints (confidence > threshold)
    valid_pts = []
    for j in range(num_joints):
        if confidences[j, 0] >= 0.01:
            valid_pts.append([int(keypoints[j, 0]), int(keypoints[j, 1])])

    if len(valid_pts) >= 3:
        pts_array = np.array(valid_pts, dtype=np.int32)

        # Convex hull → spacecraft outline
        hull = cv2.convexHull(pts_array)

        # Draw filled semi-transparent hull for subtle highlight
        overlay = vis_wire.copy()
        cv2.drawContours(overlay, [hull], 0, WIRE_COLOR, -1)
        cv2.addWeighted(overlay, 0.1, vis_wire, 0.9, 0, vis_wire)

        # Draw hull edges
        cv2.drawContours(vis_wire, [hull], 0, WIRE_COLOR, 2, cv2.LINE_AA)

        # Also connect all valid keypoints to their nearest neighbors
        # to create an internal wireframe structure
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(pts_array)
            for simplex in tri.simplices:
                for k in range(3):
                    p1 = tuple(pts_array[simplex[k]])
                    p2 = tuple(pts_array[simplex[(k + 1) % 3]])
                    cv2.line(vis_wire, p1, p2, WIRE_COLOR, 1, cv2.LINE_AA)
        except Exception:
            pass  # fallback: just use convex hull if Delaunay fails

        # Draw keypoint dots on top
        for pt in valid_pts:
            cv2.circle(vis_wire, tuple(pt), 3, WIRE_COLOR, -1, cv2.LINE_AA)

    # Save wireframe output
    base, ext = os.path.splitext(args.output)
    output_wire = base + '_wireframe' + ext
    cv2.imwrite(output_wire, vis_wire)
    print(f'Wireframe visualization saved to: {output_wire}')


if __name__ == '__main__':
    main()
