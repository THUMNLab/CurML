import math
import torch
import torch.nn as nn



######## VNet for MetaReweight, MetaWeightNet and DDS #######
class VNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class VNet_(nn.Module):
    def __init__(self, input, hidden):
        super(VNet_, self).__init__()
        self.linear1 = nn.Linear(input, hidden)

    def forward(self, x):
        x = self.linear1(x)
        return torch.sigmoid(x)

def set_parameter(current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters



######## KernelConv2d for CBS #######
class KernelConv2d(nn.Module):
    def __init__(self, conv, kernel_size, std):
        super(KernelConv2d, self).__init__()

        self.conv = conv

        self.kernel_size = kernel_size
        self.std = std
        self.channels = conv.out_channels
        self.kernel = self._get_gaussian_filter(
            self.kernel_size, self.std, self.channels
        )


    def forward(self, inputs):
        return self.kernel(self.conv(inputs))


    def model_curriculum(self, std):
        self.std = std
        self.kernel = self._get_gaussian_filter(
            self.kernel_size, self.std, self.channels
        )


    def _get_gaussian_filter(self, kernel_size, sigma, channels):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = 0

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels,
                                    bias=False, padding=padding)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter



######## SparseSGD for DataParameters ########
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
class SparseSGD(torch.optim.SGD):
    """
    This class implements SGD for optimizing parameters where at each iteration only few parameters obtain a gradient.
    More specifically, we zero out the update to state and momentum buffer for parameters with zero gradient.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        skip_update_zero_grad (bool, optional): if True, we will zero out the update to state and momentum buffer
                                                for parameters which are not in computation graph (eq. to zero gradient).
    """

    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, skip_update_zero_grad=False):
        super(SparseSGD, self).__init__(params,
                                        lr=lr,
                                        momentum=momentum,
                                        dampening=dampening,
                                        weight_decay=weight_decay,
                                        nesterov=nesterov)

        self.skip_update_zero_grad = skip_update_zero_grad
        str_disp = ' ' if self.skip_update_zero_grad else ' "not" '
        # print('Warning: skip_update_zero_grad set to {}. '
        #       'We will{}zero out update to state and momentum buffer '
        #       'for parameters with zero gradient. '.format(self.skip_update_zero_grad,
        #                                                    str_disp))
        assert weight_decay == 0, 'Weight decay for optimizer should be set to 0. ' \
                                  'For data parameters, we explicitly invoke weight decay on ' \
                                  'subset of data parameters in the computation graph.'

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Generating pointers to old-state
                p_before_update = p.data.clone()

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # Initializes momentum buffer
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                        buf_before_update = None
                    else:
                        buf = param_state['momentum_buffer']
                        buf_before_update = buf.data.clone()
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

                # We need to revert back the state of parameter and momentum buffer for entries with zero-grad
                if self.skip_update_zero_grad:
                    indices_without_grad = torch.abs(p.grad) == 0.0

                    # Old Momentum buffer has updated parameters without gradient, reverting to old value
                    p.data[indices_without_grad] = p_before_update.data[indices_without_grad]

                    # Resetting momentum buffer parameters without gradient
                    if (buf_before_update is not None) and (momentum != 0):
                        param_state['momentum_buffer'].data[indices_without_grad] = \
                            buf_before_update.data[indices_without_grad]
        return loss