from thor.data.dataset import DefaultDataset
import torchvision.transforms as transforms
from PIL import Image

class ReadRGB:
    def __call__(self, path):
        if isinstance(path, str):
            return Image.open(path).convert('RGB')
        return path

class FundusLoader(DefaultDataset):
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(768, 768), test=False):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BICUBIC)
        
        super(FundusLoader, self).__init__(data_dir, file_type, label_dir, mask_dir, target_size, test)

    def get_image_transform(self):
        transform = transforms.Compose([
            ReadRGB(),                              
            self.RES,                               
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),   
            transforms.RandomAffine(                
                degrees=15, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1)
            ),

            transforms.ToTensor(),                  
        ])
        return transform

    def get_image_transform_test(self):
        transform = transforms.Compose([
            ReadRGB(),             
            self.RES,               
            transforms.ToTensor()   
        ])
        return transform

    def get_label_transform(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])