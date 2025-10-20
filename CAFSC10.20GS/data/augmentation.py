from PIL import Image
from torchvision import transforms

class AntiNoiseAugmentation:
    """
    Anti-Noise Data Augmentation Strategy
    """
    def __init__(self, image_size=224, distortion_scale=0.3, rotation_degrees=30):
        self.image_size = image_size
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]

        self.noise_color_aug = transforms.Compose([
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=distortion_scale, p=1.0)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.3)
        ])

        self.spatial_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_degrees)
        ])

        self.final_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, pil_img: Image.Image):
        original_img = pil_img.copy()
        img_augmented = self.noise_color_aug(original_img)
        img_stitched = Image.blend(original_img, img_augmented, alpha=0.5)
        img_final = self.spatial_aug(img_stitched)
        return self.final_transform(img_final)
