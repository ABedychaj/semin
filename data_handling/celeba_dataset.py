# data/celeba_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt


class CelebAMasked(Dataset):
    def __init__(self, root_dir, img_size=128, mask_size=64, mask_type='random_rectangle', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            img_size (int): Size to resize images to
            mask_size (int): Size of the mask (for centered masks)
            mask_type (str): Type of mask - 'random_rectangle', 'center', 'random_irregular'
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.mask_size = mask_size
        self.mask_type = mask_type
        self.transform = transform or self.default_transform()

        # Get list of image files
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        image = self.transform(image)

        # Generate mask
        mask = self.generate_mask(image.shape[1:])  # Get H,W from CHW tensor

        # Apply mask
        masked_image = image * mask
        masked_image = torch.cat((masked_image, mask), dim=0)

        return masked_image, image, mask  # Return masked image, original image, and mask

    def default_transform(self):
        return T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def generate_mask(self, img_shape):
        """Generate different types of masks"""
        h, w = img_shape

        if self.mask_type == 'random_rectangle':
            # Randomly add or not add a rectangle mask
            i = np.random.randint(10)
            if i > 1:
                # Random rectangle mask
                mask = np.ones((h, w))
                y1, x1 = np.random.randint(0, h - self.mask_size), np.random.randint(0, w - self.mask_size)
                y2, x2 = y1 + self.mask_size, x1 + self.mask_size
                mask[y1:y2, x1:x2] = 0
            else:
                mask = np.ones((h, w))

        elif self.mask_type == 'center':
            # Center mask
            mask = np.ones((h, w))
            y1, x1 = (h - self.mask_size) // 2, (w - self.mask_size) // 2
            y2, x2 = y1 + self.mask_size, x1 + self.mask_size
            mask[y1:y2, x1:x2] = 0

        elif self.mask_type == 'random_irregular':
            # Random irregular mask (simplified version)
            mask = np.ones((h, w))
            for _ in range(np.random.randint(1, 2)):
                y, x = np.random.randint(0, h), np.random.randint(0, w)
                radius = np.random.randint(self.mask_size // 2, self.mask_size)
                y1 = max(0, y - radius)
                y2 = min(h, y + radius)
                x1 = max(0, x - radius)
                x2 = min(w, x + radius)
                mask[y1:y2, x1:x2] = 0

        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        return torch.from_numpy(mask).unsqueeze(0).float()  # Add channel dimension

    @staticmethod
    def denormalize(tensor):
        """Convert normalized tensor back to image format"""
        return tensor * 0.5 + 0.5  # Reverse of Normalize(mean=0.5, std=0.5)

    @staticmethod
    def plot_sample(masked_image, original_image, mask, n_samples=3):
        """
        Plot samples from the dataset
        Args:
            masked_image (Tensor): Batch of masked images
            original_image (Tensor): Batch of original images
            mask (Tensor): Batch of masks
            n_samples (int): Number of samples to plot
        """
        plt.figure(figsize=(15, 5 * n_samples))

        for i in range(n_samples):
            # Convert tensors to numpy and denormalize
            masked_img = CelebAMasked.denormalize(masked_image[i]).permute(1, 2, 0).numpy()
            orig_img = CelebAMasked.denormalize(original_image[i]).permute(1, 2, 0).numpy()
            msk = mask[i].squeeze().numpy()

            # Plot original image
            plt.subplot(n_samples, 3, i * 3 + 1)
            plt.imshow(orig_img)
            plt.title("Original Image")
            plt.axis('off')

            # Plot mask
            plt.subplot(n_samples, 3, i * 3 + 2)
            plt.imshow(msk, cmap='gray')
            plt.title("Mask")
            plt.axis('off')

            # Plot masked image
            plt.subplot(n_samples, 3, i * 3 + 3)
            plt.imshow(masked_img)
            plt.title("Masked Image")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CelebAMasked(
        root_dir="../data/celeba/img_align_celeba",
        img_size=128,
        mask_size=64,
        mask_type='random_irregular'
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch of data
    masked_images, original_images, masks = next(iter(dataloader))

    print("Masked images shape:", masked_images.shape)
    print("Original images shape:", original_images.shape)
    print("Masks shape:", masks.shape)

    # Plot samples
    CelebAMasked.plot_sample(masked_images[:4], original_images, masks, n_samples=3)
