


######## SparseSGD for DataParameters ########

# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch

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
        print('Warning: skip_update_zero_grad set to {}. '
              'We will{}zero out update to state and momentum buffer '
              'for parameters with zero gradient. '.format(self.skip_update_zero_grad,
                                                           str_disp))
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
                    d_p.add_(weight_decay, p.data)
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
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

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