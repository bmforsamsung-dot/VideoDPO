import torch


def compute_physics_weights(
    sa_scores,
    pc_scores,
    lambda_=1.0,
    kappa_alpha=8.0,
    kappa_gamma=8.0,
    b_alpha=0.5,
    b_gamma=0.5,
    alpha_min=0.05,
):
    """
    Compute physics-guided weighting terms from SA/PC scores.
    Expected SA/PC range: [0, 1].
    """
    sa_scores = torch.clamp(sa_scores, 0.0, 1.0)
    pc_scores = torch.clamp(pc_scores, 0.0, 1.0)
    v = 1.0 - 0.5 * (sa_scores + pc_scores)
    v = torch.clamp(v, 0.0, 1.0)

    alpha = alpha_min + (1.0 - alpha_min) * torch.sigmoid(
        kappa_alpha * (v - b_alpha)
    )
    gamma = 1.0 + lambda_ * torch.sigmoid(kappa_gamma * (v - b_gamma))
    return v, alpha, gamma
