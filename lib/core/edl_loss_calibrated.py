# ------------------------------------------------------------------------------
# Calibrated EDL Loss for Regression
# Based on edl_loss.py with:
#   1. Calibration loss: aligns predicted uncertainty with actual keypoint error
#   2. Temperature scaling: learnable post-hoc calibration parameter
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class NIGLossCalibrated(nn.Module):
    """
    Normal-Inverse-Gamma Loss with Calibration for Evidential Regression.

    Extends the standard NIGLoss with:
      - L_cal: calibration loss that aligns epistemic uncertainty with
               actual keypoint prediction error
      - Temperature scaling support for post-hoc uncertainty calibration

    Total loss = L_NLL + reg_coeff * L_reg + cal_coeff * L_cal
    """

    def __init__(self, reg_coeff=0.01, cal_coeff=0.05):
        """
        Args:
            reg_coeff:  coefficient for the evidence regularization term
            cal_coeff:  coefficient for the calibration loss term
        """
        super(NIGLossCalibrated, self).__init__()
        self.reg_coeff = reg_coeff
        self.cal_coeff = cal_coeff

    def nig_nll(self, y, gamma, nu, alpha, beta):
        """
        Compute NIG negative log-likelihood.

        Args:
            y:     target values,     shape (B, NUM_JOINTS)
            gamma: predicted mean,    shape (B, NUM_JOINTS)
            nu:    precision scale,   shape (B, NUM_JOINTS), > 0
            alpha: shape parameter,   shape (B, NUM_JOINTS), > 1
            beta:  rate parameter,    shape (B, NUM_JOINTS), > 0

        Returns:
            NLL loss, scalar
        """
        omega = 2.0 * beta * (1.0 + nu)

        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        return nll.mean()

    def nig_reg(self, y, gamma, nu, alpha):
        """
        Evidence regularizer: penalizes evidence when prediction is wrong.

        Args:
            y:     target values,     shape (B, NUM_JOINTS)
            gamma: predicted mean,    shape (B, NUM_JOINTS)
            nu:    precision scale,   shape (B, NUM_JOINTS)
            alpha: shape parameter,   shape (B, NUM_JOINTS)

        Returns:
            Regularization loss, scalar
        """
        reg = torch.abs(y - gamma) * (2.0 * nu + alpha)
        return reg.mean()

    def calibration_loss(self, pred_coords, gt_coords, nu, alpha, beta):
        """
        Calibration loss: aligns epistemic uncertainty with actual prediction error.

        The idea: the predicted uncertainty should correlate with the actual error.
        If a keypoint has high prediction error, the uncertainty should also be high.

        L_cal = sum_j (log(sigma2_j) - log(error_j^2 + eps))^2

        where sigma2_j = beta / (nu * (alpha - 1))  is the epistemic uncertainty,
        and error_j = |pred_j - gt_j| is the actual prediction error.

        Args:
            pred_coords:  predicted keypoint heatmap peaks, shape (B, NUM_JOINTS)
            gt_coords:    GT keypoint heatmap peaks,        shape (B, NUM_JOINTS)
            nu:           precision scale,                  shape (B, NUM_JOINTS)
            alpha:        shape parameter,                  shape (B, NUM_JOINTS)
            beta:         rate parameter,                   shape (B, NUM_JOINTS)

        Returns:
            Calibration loss, scalar
        """
        eps = 1e-8

        # Epistemic uncertainty from NIG
        sigma2 = beta / (nu * torch.clamp(alpha - 1.0, min=eps))

        # Actual prediction error (squared)
        error_sq = (pred_coords - gt_coords) ** 2 + eps

        # Log-space alignment: encourages log(sigma2) ≈ log(error^2)
        log_sigma2 = torch.log(sigma2 + eps)
        log_error = torch.log(error_sq)

        cal_loss = (log_sigma2 - log_error) ** 2

        return cal_loss.mean()

    def forward(self, nig_params, target_coords, pred_coords=None, target_weight=None):
        """
        Compute the total calibrated EDL loss.

        Args:
            nig_params:     NIG parameters from model, shape (B, NUM_JOINTS, 4)
                            where last dim is [gamma, nu, alpha, beta]
            target_coords:  GT keypoint heatmap peaks,
                            shape (B, NUM_JOINTS)
            pred_coords:    predicted keypoint heatmap peaks (for calibration),
                            shape (B, NUM_JOINTS). If None, calibration loss is skipped.
            target_weight:  optional weight per joint, shape (B, NUM_JOINTS)
                            1.0 for visible joints, 0.0 for invisible

        Returns:
            total loss (scalar)
        """
        gamma = nig_params[:, :, 0]   # (B, NUM_JOINTS)
        nu    = nig_params[:, :, 1]   # (B, NUM_JOINTS)
        alpha = nig_params[:, :, 2]   # (B, NUM_JOINTS)
        beta  = nig_params[:, :, 3]   # (B, NUM_JOINTS)

        nll = self.nig_nll(target_coords, gamma, nu, alpha, beta)
        reg = self.nig_reg(target_coords, gamma, nu, alpha)

        loss = nll + self.reg_coeff * reg

        # Add calibration loss if predicted coordinates are provided
        if pred_coords is not None and self.cal_coeff > 0:
            cal = self.calibration_loss(pred_coords, target_coords, nu, alpha, beta)
            loss = loss + self.cal_coeff * cal

        return loss

    @staticmethod
    def compute_uncertainty(nig_params, temperature=1.0):
        """
        Compute epistemic uncertainty from NIG parameters with temperature scaling.

        Epistemic uncertainty (model uncertainty) = T * beta / (nu * (alpha - 1))

        Temperature scaling:
          T > 1: increases uncertainty (less confident, more conservative)
          T < 1: decreases uncertainty (more confident)
          T = 1: no scaling (original EDL output)

        Args:
            nig_params:  shape (B, NUM_JOINTS, 4) — [gamma, nu, alpha, beta]
            temperature: scalar temperature parameter (default: 1.0)

        Returns:
            uncertainty: shape (B, NUM_JOINTS), calibrated epistemic uncertainty
        """
        nu    = nig_params[:, :, 1]
        alpha = nig_params[:, :, 2]
        beta  = nig_params[:, :, 3]

        # Clamp (alpha - 1) to avoid division by zero
        epistemic = beta / (nu * torch.clamp(alpha - 1.0, min=1e-6))

        # Apply temperature scaling
        epistemic = temperature * epistemic

        return epistemic
