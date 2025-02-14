import torch
import torch.nn.functional as F


def gradient_penalty(net, x, x_hat):

        alpha = torch.rand(x.size(0)).unsqueeze(1).unsqueeze(2).to(x.device)
        #set_trace()
        interpolated = x * alpha + x_hat * (1 - alpha)
        interpolated = torch.autograd.Variable(interpolated.detach(), requires_grad=True)
        output = net(interpolated )
        gradients = torch.autograd.grad(output,
                                        interpolated,
                                        torch.ones_like(output),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)[0]
        penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()             # one-sided
        # penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()          # two-sided
        return penalty
