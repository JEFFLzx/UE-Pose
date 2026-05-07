# ------------------------------------------------------------------------------
# EDL (Evidential Deep Learning) Training and Validation Functions
# Based on function.py but replacing MC Dropout with single-pass EDL uncertainty
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from core.edl_loss import NIGLoss
from utils.transforms import flip_back
from utils.vis import save_debug_images_sensor_fusion
from kornia.geometry.conversions import axis_angle_to_rotation_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

logger = logging.getLogger(__name__)


def _extract_keypoint_coords_from_heatmap(heatmaps):
    """
    Extract 2D keypoint coordinates from heatmaps by taking argmax.
    Used as the 'target' for EDL uncertainty head: the uncertainty head
    learns to predict a scalar value per joint that can be compared against
    the heatmap peak magnitude.

    Args:
        heatmaps: (B, NUM_JOINTS, H, W) tensor (GT or predicted heatmaps)

    Returns:
        peak_vals: (B, NUM_JOINTS) tensor — the peak value of each heatmap
    """
    B, J, H, W = heatmaps.shape
    hm_flat = heatmaps.view(B, J, -1)                  # (B, J, H*W)
    peak_vals, _ = hm_flat.max(dim=-1)                  # (B, J)
    return peak_vals


def train_posehrnetfusion_nofusion_edl(config, 
                        train_loader_event_rgb, 
                        model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb,
                        heatmap_loss_event, heatmap_loss_rgb,
                        edl_loss_event, edl_loss_rgb,
                        optimizer_event, optimizer_rgb,
                        epoch, output_dir, tb_log_dir, writer_dict,
                        edl_lambda=0.1):
    """
    Training function for EDL-based uncertainty estimation.
    Joint training: heatmap MSE loss + lambda * EDL NIG loss.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_event = AverageMeter()
    losses_edl_event = AverageMeter()
    acc_event = AverageMeter()
    losses_rgb = AverageMeter()
    losses_edl_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    print("TRAINING NO FUSION - EDL UNCERTAINTY")

    end = time.time()
    for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(train_loader_event_rgb):
        data_time.update(time.time() - end)

        # ---- EVENT CHANNEL ----
        model_hrnet_encoder_event.train()
        model_event.train()
        
        features_event = model_hrnet_encoder_event(input_tensor_event)
        outputs_event = model_event(features_event)
        
        # EDL mode returns (heatmap, nig_params)
        if isinstance(outputs_event, tuple):
            output_event, nig_params_event = outputs_event
        else:
            output_event = outputs_event
            nig_params_event = None
        
        target_event = target_event.cuda(non_blocking=True)
        target_weight_event = target_weight_event.cuda(non_blocking=True)

        loss_hm_event = heatmap_loss_event(output_event, target_event, target_weight_event)
        
        # EDL loss: target is the peak value of GT heatmap per joint
        loss_edl_ev = torch.tensor(0.0, device=output_event.device)
        if nig_params_event is not None and edl_loss_event is not None:
            target_coords_event = _extract_keypoint_coords_from_heatmap(target_event)
            loss_edl_ev = edl_loss_event(nig_params_event, target_coords_event)
        
        total_loss_event = loss_hm_event + edl_lambda * loss_edl_ev
        
        optimizer_event.zero_grad()
        total_loss_event.backward()
        optimizer_event.step()
        
        model_hrnet_encoder_event.eval()
        model_event.eval()
        
        losses_event.update(loss_hm_event.item(), input_tensor_event.size(0))
        losses_edl_event.update(loss_edl_ev.item(), input_tensor_event.size(0))

        _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                               target_event.detach().cpu().numpy())
        acc_event.update(avg_acc, cnt)
        
        # ---- RGB CHANNEL ----
        model_hrnet_encoder_rgb.train()
        model_rgb.train()
        
        features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
        outputs_rgb = model_rgb(features_rgb)
        
        if isinstance(outputs_rgb, tuple):
            output_rgb, nig_params_rgb = outputs_rgb
        else:
            output_rgb = outputs_rgb
            nig_params_rgb = None

        target_rgb = target_rgb.cuda(non_blocking=True)
        target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

        loss_hm_rgb = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)
        
        loss_edl_r = torch.tensor(0.0, device=output_rgb.device)
        if nig_params_rgb is not None and edl_loss_rgb is not None:
            target_coords_rgb = _extract_keypoint_coords_from_heatmap(target_rgb)
            loss_edl_r = edl_loss_rgb(nig_params_rgb, target_coords_rgb)
        
        total_loss_rgb = loss_hm_rgb + edl_lambda * loss_edl_r
        
        optimizer_rgb.zero_grad()
        total_loss_rgb.backward()
        optimizer_rgb.step()
        
        model_hrnet_encoder_rgb.eval()
        model_rgb.eval()

        losses_rgb.update(loss_hm_rgb.item(), input_tensor_rgb.size(0))
        losses_edl_rgb.update(loss_edl_r.item(), input_tensor_rgb.size(0))

        _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                             target_rgb.detach().cpu().numpy())
        acc_rgb.update(avg_acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_event {loss_event.val:.7f} ({loss_event.avg:.7f})\t' \
                  'EDL_event {edl_event.val:.7f} ({edl_event.avg:.7f})\t' \
                  'Loss_rgb {loss_rgb.val:.7f} ({loss_rgb.avg:.7f})\t' \
                  'EDL_rgb {edl_rgb.val:.7f} ({edl_rgb.avg:.7f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader_event_rgb), batch_time=batch_time,
                      speed=input_tensor_rgb.size(0)/batch_time.val,
                      data_time=data_time, 
                      loss_event=losses_event, edl_event=losses_edl_event,
                      loss_rgb=losses_rgb, edl_rgb=losses_edl_rgb,
                      acc=acc_event)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_event', losses_event.val, global_steps)
            writer.add_scalar('train_edl_loss_event', losses_edl_event.val, global_steps)
            writer.add_scalar('train_acc_event', acc_event.val, global_steps)
            writer.add_scalar('train_loss_rgb', losses_rgb.val, global_steps)
            writer.add_scalar('train_edl_loss_rgb', losses_edl_rgb.val, global_steps)
            writer.add_scalar('train_acc_rgb', acc_rgb.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_debug_images_sensor_fusion(
                config, 
                input_tensor_event, input_tensor_rgb, 
                meta, 
                pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), prefix)


def validate_posehrnetfusion_nofusion_edl(
    config, 
    val_loader, 
    val_dataset, 
    model_event, model_rgb, model_hrnet_encoder_event, model_hrnet_encoder_rgb, 
    heatmap_loss_event, heatmap_loss_rgb, 
    output_dir, tb_log_dir, writer_dict=None, epoch=0):
    """
    Validation function with EDL uncertainty — single forward pass per sample.
    No need for MC Dropout loops: uncertainty comes directly from NIG parameters.
    """
    batch_time = AverageMeter()
    losses_event = AverageMeter()
    acc_event = AverageMeter()
    losses_rgb = AverageMeter()
    acc_rgb = AverageMeter()

    # switch to evaluate mode
    model_event.eval()
    model_rgb.eval()
    model_hrnet_encoder_event.eval()
    model_hrnet_encoder_rgb.eval()

    num_samples = len(val_dataset)
    
    # Single-pass predictions (no need for uncertainty_K dimension)
    keypoint_predictions_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 1, 3),
        dtype=np.float32
    )
    keypoint_variances_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_means_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_covs_event = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2, 2),
        dtype=np.float32
    )
    pose_predictions_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_event = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_event = []
    
    keypoint_predictions_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 1, 3),
        dtype=np.float32
    )
    keypoint_variances_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_means_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2),
        dtype=np.float32
    )
    keypoint_covs_rgb = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 2, 2),
        dtype=np.float32
    )
    pose_predictions_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    pose_gt_rgb = np.tile(np.eye(4, dtype=np.float32), (num_samples, 1, 1))
    filenames_rgb = []

    idx = 0
    
    with torch.no_grad():
        end = time.time()
        for i, (input_tensor_event, target_event, target_weight_event, trans_hm_to_img_event, trans_crop_to_img_event, input_tensor_rgb, target_rgb, target_weight_rgb, trans_hm_to_img_rgb, trans_crop_to_img_rgb, meta) in enumerate(val_loader):
            
            # ---- EVENT CHANNEL (single forward pass) ----
            input_tensor_event = input_tensor_event.cuda()
            features_event = model_hrnet_encoder_event(input_tensor_event)
            
            outputs_event = model_event(features_event)
            if isinstance(outputs_event, tuple):
                output_event, nig_params_event = outputs_event
            else:
                output_event = outputs_event
                nig_params_event = None

            target_event = target_event.cuda(non_blocking=True)
            target_weight_event = target_weight_event.cuda(non_blocking=True)

            loss = heatmap_loss_event(output_event, target_event, target_weight_event)

            num_images = input_tensor_event.size(0)
            losses_event.update(loss.item(), num_images)
            _, avg_acc, cnt, pred_event = accuracy(output_event.detach().cpu().numpy(),
                                             target_event.detach().cpu().numpy())
            acc_event.update(avg_acc, cnt)
            
            centre_event = meta['center_event'].numpy()
            scale_event = meta['scale_event'].numpy()

            preds_event, maxvals = get_final_preds(
                config, output_event.clone().cpu().numpy(), centre_event, scale_event)

            # Store single-pass predictions (uncertainty_K=0)
            keypoint_predictions_event[idx:idx + num_images, :, 0, 0:2] = preds_event[:, :, 0:2]
            keypoint_predictions_event[idx:idx + num_images, :, 0, 2:3] = maxvals
            
            # EDL uncertainty as variance proxy
            if nig_params_event is not None:
                epistemic_event = NIGLoss.compute_uncertainty(nig_params_event)  # (B, NUM_JOINTS)
                unc_event_np = epistemic_event.cpu().numpy()
                # Use the same uncertainty for both x and y
                keypoint_variances_event[idx:idx + num_images, :, 0] = unc_event_np
                keypoint_variances_event[idx:idx + num_images, :, 1] = unc_event_np
            
            keypoint_means_event[idx:idx + num_images, :, 0:2] = preds_event[:, :, 0:2]
            
            filenames_event.extend(meta['image_filename_event'])
            
            # get the event poses
            keypoints_bpnp_event, poses_bpnp_event, _ = val_dataset.pose_estimator_event.predict(output_event.clone(), trans_hm_to_img_event)
            poses_bpnp_event = poses_bpnp_event.clone()
            pose_predictions_event[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_event[:, 0:3]).cpu().numpy()
            poses_bpnp_event = poses_bpnp_event.cpu().numpy()
            pose_predictions_event[idx:idx + num_images, 0:3, -1] = poses_bpnp_event[:, 3:].reshape((-1, 3))
            
            pose_gt_event[idx:idx + num_images, :, :] = meta["pose_event"]
            
            # ---- RGB CHANNEL (single forward pass) ----
            input_tensor_rgb = input_tensor_rgb.cuda()
            features_rgb = model_hrnet_encoder_rgb(input_tensor_rgb)
            
            outputs_rgb = model_rgb(features_rgb)
            if isinstance(outputs_rgb, tuple):
                output_rgb, nig_params_rgb = outputs_rgb
            else:
                output_rgb = outputs_rgb
                nig_params_rgb = None

            target_rgb = target_rgb.cuda(non_blocking=True)
            target_weight_rgb = target_weight_rgb.cuda(non_blocking=True)

            loss = heatmap_loss_rgb(output_rgb, target_rgb, target_weight_rgb)

            num_images = input_tensor_rgb.size(0)
            losses_rgb.update(loss.item(), num_images)
            _, avg_acc, cnt, pred_rgb = accuracy(output_rgb.detach().cpu().numpy(),
                                             target_rgb.detach().cpu().numpy())
            acc_rgb.update(avg_acc, cnt)

            centre_rgb = meta['center_rgb'].numpy()
            scale_rgb = meta['scale_rgb'].numpy()

            preds_rgb, maxvals = get_final_preds(
                config, output_rgb.clone().cpu().numpy(), centre_rgb, scale_rgb)

            keypoint_predictions_rgb[idx:idx + num_images, :, 0, 0:2] = preds_rgb[:, :, 0:2]
            keypoint_predictions_rgb[idx:idx + num_images, :, 0, 2:3] = maxvals
            
            if nig_params_rgb is not None:
                epistemic_rgb = NIGLoss.compute_uncertainty(nig_params_rgb)  # (B, NUM_JOINTS)
                unc_rgb_np = epistemic_rgb.cpu().numpy()
                keypoint_variances_rgb[idx:idx + num_images, :, 0] = unc_rgb_np
                keypoint_variances_rgb[idx:idx + num_images, :, 1] = unc_rgb_np
            
            keypoint_means_rgb[idx:idx + num_images, :, 0:2] = preds_rgb[:, :, 0:2]
            
            filenames_rgb.extend(meta['image_filename_rgb'])
            
            # get the rgb poses
            keypoints_bpnp_rgb, poses_bpnp_rgb, _ = val_dataset.pose_estimator_rgb.predict(output_rgb.clone(), trans_hm_to_img_rgb)
            poses_bpnp_rgb = poses_bpnp_rgb.clone()
            pose_predictions_rgb[idx:idx + num_images, 0:3, 0:3] = axis_angle_to_rotation_matrix(poses_bpnp_rgb[:, 0:3]).cpu().numpy()
            poses_bpnp_rgb = poses_bpnp_rgb.cpu().numpy()
            pose_predictions_rgb[idx:idx + num_images, 0:3, -1] = poses_bpnp_rgb[:, 3:].reshape((-1, 3))
            
            pose_gt_rgb[idx:idx + num_images, :, :] = meta["pose_rgb"]
            
            idx += num_images
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Testevent: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_event, acc=acc_event)
                logger.info(msg)
                msg = 'Testrgb: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_rgb, acc=acc_rgb)
                logger.info(msg)

                prefix = '{}_{}_{}'.format(
                    os.path.join(output_dir, 'validation'), epoch, i
                )
                save_debug_images_sensor_fusion(
                    config, 
                    input_tensor_event, 
                    input_tensor_rgb, meta, 
                    pred_event*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    pred_rgb*(config.MODEL.IMAGE_SIZE[0]/float(config.MODEL.HEATMAP_SIZE[0])), 
                    prefix)
                

        perf_indicator_event, perf_indicator_rgb = val_dataset.evaluate_with_uncertainty(
            config, 
            output_dir,
            pose_gt_event,
            keypoint_predictions_event,
            keypoint_variances_event,
            keypoint_means_event,
            keypoint_covs_event,
            pose_predictions_event,
            filenames_event,
            pose_gt_rgb,
            keypoint_predictions_rgb, 
            keypoint_variances_rgb,
            keypoint_means_rgb,
            keypoint_covs_rgb,
            pose_predictions_rgb,
            filenames_rgb
        )

    return perf_indicator_event, perf_indicator_rgb
