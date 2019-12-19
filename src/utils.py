import torch
from torch import nn


class LReluCustom(nn.Module):
    def __init__(self, leak=0.1):
        super(LReluCustom, self).__init__()
        self.leak = leak

    def forward(self, x):
        return torch.max(x, self.leak * x)


def softmax_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    eps: float = 1e-20
):
    uniform = torch.rand(logits.shape, device=logits.device)
    noise = -torch.log(-torch.log(uniform + eps) + eps)
    y = logits + noise
    return nn.functional.softmax(y / temperature, dim=-1)


def generate_inter_sample(
    fake: torch.Tensor,
    real: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        alpha = torch.rand([real.shape[0], 1, 1], device=fake.device)
        inter_sample = real + alpha * (fake - real)
        return inter_sample


# gradients = tf.gradients(inter_sample_pred, [inter_sample])[0]
# slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
# reduction_indices=[1,2]))
# return tf.reduce_mean((slopes-1.)**2)
def compute_gradient_penalty(output, input):
    # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master
    # /implementations/wgan_gp/wgan_gp.py
    '''

    :param input: state[index]
    :param network: actor or critic
    :return: gradient penalty
    '''
    # TODO Fix This shiit. None value returned
    input_ = torch.tensor(input).requires_grad_(True)
    print(output)
    musk = torch.ones_like(output).requires_grad(True)
    gradients = torch.autograd.grad(
        output,
        input_,
        grad_outputs=musk,
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]  # get tensor from tuple
    breakpoint()
    gradients = gradients.view(-1, 1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
