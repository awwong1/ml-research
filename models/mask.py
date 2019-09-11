import torch


class BinaraizerSTEStatic(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor. Backward pass is a Straight Through Estimator"""

    @staticmethod
    def forward(ctx, threshold, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad):
        return None, grad.clone()


class MaskSTE(torch.nn.Module):
    """Apply a differentiable mask on the input."""

    def __init__(
        self,
        mask_size,
        kernel_size=1,
        in_channels=1,
        mask_scale=1e-2,
        apply_sigmoid=True,
        threshold=0.5,
    ):
        super(MaskSTE, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.thresholdfn = BinaraizerSTEStatic.apply
        self.threshold = threshold

        # initialize real value mask
        self.mask_real = torch.rand_like(torch.FloatTensor(size=mask_size)).mul(
            mask_scale
        )
        # self.mask_real.add_(torch.ones_like(self.mask_real).mul(mask_scale))
        self.mask_real = torch.nn.Parameter(self.mask_real, requires_grad=True)
        self.factor = kernel_size if len(kernel_size) == 1 else kernel_size[0] * kernel_size[1]
        self.factor = self.factor * in_channels
        # self.factor = torch.nn.Parameter(self.factor, requires_grad=False)

    def __repr__(self):
        return "{n}(mask_size={ms}, factor={f}, apply_sigmoid={s}, threshold={t})".format(
            n=self.__class__.__name__,
            ms=tuple(self.mask_real.size()),
            f=self.factor,
            s=self.apply_sigmoid,
            t=self.threshold
        )

    def forward(self, input_feature):
        mask_thresholded, _ = self.get_binary_mask()
        return input_feature * mask_thresholded

    def get_binary_mask(self):
        mask_real_out = self.mask_real
        if self.apply_sigmoid:
            mask_real_out = torch.nn.Sigmoid()(self.mask_real)
        mask_thresholded = self.thresholdfn(self.threshold, mask_real_out)
        return mask_thresholded, self.factor
