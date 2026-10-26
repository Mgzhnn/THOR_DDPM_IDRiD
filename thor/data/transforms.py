import numpy as np
import torch
import PIL
from monai.transforms import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
import torchvision.transforms.functional as transform


class ReadImage(Transform):
    """
    Transform to read image, see torchvision.io.image.read_image
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, path: str) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if '.npy' in path:
            img = np.load(path).astype(np.float32)
            img = (img * 255).astype(np.uint8)
            return torch.tensor(img)
        elif '.jpeg' in path or '.jpg' in path or '.png' in path:
            PIL_image = PIL.Image.open(path)
            return torch.squeeze(transform.to_tensor(PIL_image))
        else:
            raise IOError


class To01:
    """
    Convert the input to [0,1] scale

    """
    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(To01, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        if torch.max(img) <= 1.0:
            return img
        if torch.max(img) <= 255.0:
            return img / 255

        return img / 65536


class AddChannelIfNeeded(Transform):
    """
    Adds a 1-length channel dimension to the input image, if input is 2D
    """
    def __init__(self, dim=2):
        self.dim=dim

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if (self.dim == 2 and len(img.shape) == 2) or (self.dim == 3 and len(img.shape) == 3):
            return img[None, ...]
        else:
            return img


class AssertChannelFirst(Transform):
    """
    Assert channel is first and permute otherwise
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        assert len(img.shape) == 3,  f'AssertChannelFirst:: Image should have 3 dimensions, instead of {len(img.shape)}'
        if img.shape[0] == img.shape[1] and img.shape[0] != img.shape[2]:
            print(f'Permuted channels {(img.permute(2,0,1)).shape}')
            return img.permute(2, 0, 1)
        elif img.shape[0] > 1:
            return img[0: 1, :, :]
        else:
            return img
