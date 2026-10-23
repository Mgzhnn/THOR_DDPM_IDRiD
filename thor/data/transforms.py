import numpy as np
import torch
import PIL
# Reads a file using pillow
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
            # return torch.tensor(np.flipud((np.load(path).astype(np.float32)).T).copy())  # Mid Axial slice of MRI brains
            img = np.load(path).astype(np.float32)
            img = (img * 255).astype(np.uint8)
            return torch.tensor(img)
        elif '.jpeg' in path or '.jpg' in path or '.png' in path:
            PIL_image = PIL.Image.open(path)#.convert('L')
            # The image can be converted to tensor using
            tensor_image = torch.squeeze(transform.to_tensor(PIL_image))
            # print(f'Read:: Min: {torch.min(tensor_image)}, Max: {torch.max(tensor_image)}, Avg: '
            #       f'{torch.mean(tensor_image)}')

            # print(tensor_image.cpu().numpy().shape)
            return tensor_image#read_image(path)
        elif '.nii.gz' in path:
            import nibabel as nip
            from nibabel.imageglobals import LoggingOutputSuppressor
            with LoggingOutputSuppressor():
                img_obj = nip.load(path)
                img_np = np.array(img_obj.get_fdata(), dtype=np.float32)
                # img_t = torch.Tensor(img_np.copy())
                img_t = torch.Tensor(img_np[:, :, :].copy()) # Transposed 3D MRI brains
                # img_t = torch.Tensor(np.flipud(img_np[:, :, :].T).copy()) # Transposed 3D MRI brains
                # img_t = torch.Tensor(np.flipud(img_np[:, :, 95].T).copy()) # Mid Axial slice of MRI brains
            # TODO: THIS IS TEMPORARY FI IT
            # if img_t.shape[2] > 1:
            #     mid_slice = int(img_t.shape[2]/2)
            #     img_t = img_t[:,:,mid_slice:mid_slice+1]
            #     img_t = img_t.permute(2,0,1)
            return img_t
        elif '.nii' in path:
            import nibabel as nip
            img = nip.load(path)
            return torch.Tensor(np.array(img.get_fdata()))
        elif '.dcm' in path:
            from pydicom import dcmread
            ds = dcmread(path)
            return torch.Tensor(ds.pixel_array)
        elif '.h5' in path:  ## !!! SPECIFIC TO FAST MRI, CHANGE FOR OTHER DATASETS
            import h5py
            f = h5py.File(path, 'r')
            img_data = f['reconstruction_rss'][:] # Fast MRI Specific
            img_data = img_data[:, ::-1, :][0]  # flipped up down
            return torch.tensor(img_data.copy())
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
        # print(torch.max(img))
        # print(f'To01:: Min: {torch.min(img)}, Max: {torch.max(img)}, Avf: {torch.mean(img)}')
        # print(f'Input image shape: {img.shape}')
        # print(f'Max_val  {self.max_val}:: Min: {torch.min(img)}, Max: {torch.max(img)}, Avg: {torch.mean(img)}')
        if torch.max(img) <= 1.0:
            # print(img.cpu().numpy().shape)
            return img
        # print(img.cpu().numpy().shape)
        if torch.max(img) <= 255.0:
            return img / 255

        # return img / torch.max(img)
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
        # print(torch.min(img), torch.max(img))
        # print(f'AssertCH:: Min: {torch.min(img)}, Max: {torch.max(img)}, Avg: {torch.mean(img)}')
        assert len(img.shape) == 3,  f'AssertChannelFirst:: Image should have 3 dimensions, instead of {len(img.shape)}'
        if img.shape[0] == img.shape[1] and img.shape[0] != img.shape[2]:
            print(f'Permuted channels {(img.permute(2,0,1)).shape}')
            return img.permute(2, 0, 1)
        elif img.shape[0] > 1:
            # print(img.shape)
            return img[0: 1, :, :]
        else:
            # print(img.shape)
            return img


