# ------------------------------------------------------------------------------
# Evidential Deep Learning (EDL) Loss for Regression
# Based on: "Deep Evidential Regression" (Amini et al., NeurIPS 2020)
# Normal-Inverse-Gamma (NIG) negative log-likelihood + evidence regularizer
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class NIGLoss(nn.Module):
    """
    Normal-Inverse-Gamma Negative Log-Likelihood Loss for Evidential Regression.

    Given NIG parameters (gamma, nu, alpha, beta) and target y, computes:
        L_NLL = 0.5 * log(pi / nu)
              - alpha * log(Omega)
              + (alpha + 0.5) * log((y - gamma)^2 * nu + Omega)
              + log(Gamma(alpha) / Gamma(alpha + 0.5))
        where Omega = 2 * beta * (1 + nu)

    Plus an evidence regularizer:
        L_reg = |y - gamma| * (2 * nu + alpha)

    Total loss = L_NLL + coeff * L_reg
    """

    def __init__(self, reg_coeff=0.01):
        """
        Args:
            reg_coeff: coefficient for the evidence regularization term
        """
        super(NIGLoss, self).__init__()
        self.reg_coeff = reg_coeff

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

    def forward(self, nig_params, target_coords, target_weight=None):
        """
        Compute the total EDL loss.

        Args:
            nig_params:     NIG parameters from model, shape (B, NUM_JOINTS, 4)
                            where last dim is [gamma, nu, alpha, beta]
            target_coords:  GT keypoint coordinates (e.g. from heatmap argmax),
                            shape (B, NUM_JOINTS) — we use the trace/magnitude
                            as a scalar target for each keypoint
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

        return loss

    @staticmethod
    def compute_uncertainty(nig_params):
        """
        Compute epistemic uncertainty from NIG parameters.

        Epistemic uncertainty (model uncertainty) = beta / (nu * (alpha - 1))

        Args:
            nig_params: shape (B, NUM_JOINTS, 4) — [gamma, nu, alpha, beta]

        Returns:
            uncertainty: shape (B, NUM_JOINTS), epistemic uncertainty per joint
        """
        nu    = nig_params[:, :, 1]
        alpha = nig_params[:, :, 2]
        beta  = nig_params[:, :, 3]

        # Clamp (alpha - 1) to avoid division by zero
        epistemic = beta / (nu * torch.clamp(alpha - 1.0, min=1e-6))

        return epistemic
