import torch
import torch.nn.functional as F
from thor.models.vgg import VGGEncoder
from torch.nn.modules.loss import _Loss


class PerceptualLoss(_Loss):
    """
    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = VGGEncoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        # VGG takes 3 channels: repeat grayscale, pass RGB through (RGB previously skipped VGG and compared raw pixels)
        input_features = self.loss_network(input.repeat(1, 3, 1, 1) if input.shape[1] == 1 else input)
        output_features = self.loss_network(target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target)

        loss_pl = 0
        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
        return loss_pl
