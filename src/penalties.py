import torch
from torch.autograd import (
    Variable,
    grad,
)

from src.discriminator_net import DiscriminatorNet
from src.utils import generate_inter_sample


def compute_gradient_penalty(
    discriminator: DiscriminatorNet,
    real_data: torch.tensor,
    generated_data: torch.tensor
):
    # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master
    # /implementations/wgan_gp/wgan_gp.py
    '''

    :param input: state[index]
    :param network: actor or critic
    :return: gradient penalty
    '''
    batch_size = real_data.size()[0]
    device = generated_data.device
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(
        interpolated,
        requires_grad=True
    ).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator.forward(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(
            prob_interpolated.size()
        ).to(device),
        create_graph=True,
        retain_graph=True
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()
