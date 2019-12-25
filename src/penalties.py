import torch

from src.discriminator_net import DiscriminatorNet
from src.utils import generate_inter_sample


def compute_gradient_penalty(
    discriminator: DiscriminatorNet,
    real_sample: torch.tensor,
    fake_sample: torch.tensor
):
    # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master
    # /implementations/wgan_gp/wgan_gp.py
    '''

    :param input: state[index]
    :param network: actor or critic
    :return: gradient penalty
    '''
    inter_sample = generate_inter_sample(
        fake_sample,
        real_sample
    ).requires_grad_(True)

    inter_sample_pred = discriminator.forward(inter_sample)

    fake = torch.ones_like(inter_sample_pred, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=inter_sample_pred,
        inputs=inter_sample,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
