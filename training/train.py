import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from utils.logger import Logger


class Trainer:
    def __init__(self, model, train_loader, config, optimizer=None):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr']) if optimizer is None else optimizer
        self.logger = Logger(config['save_dir'])
        self.step = 0

        # Diffusion parameters
        self.beta = torch.linspace(0.0001, 0.02, self.config["timesteps"]).to(self.config['device'])
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        os.makedirs(config['save_dir'], exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        with tqdm(self.train_loader, desc="Training", leave=True) as pbar:
            for batch in self.train_loader:
                masked_images, real_images, masks = batch
                masked_images = masked_images.to(self.config['device'])
                real_images = real_images.to(self.config['device'])

                # Forward pass
                self.optimizer.zero_grad()
                loss = self.train_step(masked_images, real_images)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                self.step += 1

                # Logging
                if self.step % 10 == 0:
                    self.logger.log({
                        'loss': total_loss / (self.step + 1),
                        'step': self.step
                    })
                    pbar.set_description(f"Training (loss: {total_loss / (self.step + 1):.4f})")

                # Save samples periodically
                if self.step % 1000 == 0:
                    self.save_samples(masked_images[:2], real_images[:2])

                pbar.update(1)

        return total_loss / len(self.train_loader)

    def train_step(self, masked_img, real_img):
        # Generate random timesteps
        batch_size = real_img.size(0)
        t = torch.randint(0, self.config["timesteps"], (batch_size,), device=self.config['device']).long()

        # Add noise to real images
        noise = torch.randn_like(real_img)
        noisy_img = self.q_sample(real_img, t, noise)

        # Generate conditioning if needed
        cond = masked_img if self.model.cond_dim is not None else None

        # Model prediction
        pred_noise = self.model(noisy_img, t, cond)

        # Loss calculation
        loss = (torch.nn.functional.mse_loss(pred_noise, noise, reduction='none') * cond[:, -1:]).sum() / cond[:,
                                                                                                          -1:].sum()
        return loss

    def q_sample(self, x_start, t, noise):
        # Compute alpha_bar for current timesteps
        cur_alpha_bar = self.alpha_bar[t][:, None, None, None]

        return torch.sqrt(cur_alpha_bar) * x_start + \
            torch.sqrt(1 - cur_alpha_bar) * noise

    def reverse_diffusion_step(self, x_t, t, pred_noise):
        """Performs single reverse diffusion step from x_t to x_{t-1}

        Args:
            x_t: Noisy input at timestep t (batch_size × channels × H × W)
            t: Current timestep values for each sample (batch_size,)
            pred_noise: Model's noise prediction (same shape as x_t)

        Returns:
            x_prev: Denoised sample at timestep t-1
        """
        # Extract schedule parameters with dimension broadcasting
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # Calculate mean component of reverse process
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        # Compute x_{t-1} mean using noise prediction
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise) / sqrt_alpha_t

        # Calculate variance component only for t > 0
        if t[0] > 0:  # Check if any in batch require noise
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(beta_t) * noise
        else:
            variance = 0

        # Combine mean and variance
        x_prev = mean + variance

        return x_prev

    def sample_diffusion(self, model, initial_noise, num_steps, cond, device):
        """
        Generates samples by reversing the diffusion process.

        Args:
            model (torch.nn.Module): Trained neural network for noise prediction.
            initial_noise (torch.Tensor): Starting point (e.g., Gaussian noise).
            betas (torch.Tensor): Noise variance schedule.
            num_steps (int): Number of diffusion steps.
            device (torch.device): Device for computation.

        Returns:
            torch.Tensor: Generated sample after denoising.
        """
        x = initial_noise.to(device)
        for t in reversed(range(num_steps)):
            beta_t = self.beta[t]
            predicted_noise = model(x, torch.tensor([t], device=device), cond)  # Predict noise at step t
            x = (x - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(1 - beta_t)  # Reverse diffusion step

            # Optional: Clamp values to prevent overflow, ensuring pixel values remain within a realistic range
            x = torch.clamp(x, 0.0, 1.0)

        return x

    def save_samples(self, masked, real):
        with torch.no_grad():
            reconstructed = self.sample_diffusion(self.model, torch.randn_like(real), 50, masked,
                                                  self.config['device'])

            # Denormalize images
            masked = self.denormalize(masked[:, :3])
            real = self.denormalize(real)
            reconstructed = self.denormalize(reconstructed)

            # Create grid
            comparison = torch.cat([masked, reconstructed, real], dim=0)
            save_path = os.path.join(self.config['save_dir'], f'samples_step_{self.step}.png')
            save_image(comparison, save_path, nrow=masked.size(0))

    def denormalize(self, tensor):
        return tensor.cpu() * 0.5 + 0.5

    def train(self, epoch):
        for epoch in range(epoch, self.config['epochs']):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{self.config['epochs']} | Loss: {avg_loss:.4f}")

            ckpt_path = os.path.join(self.config['save_dir'], f'model_epoch_latest.pth')
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch + 1
            }, ckpt_path)

            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(self.config['save_dir'], f'model_epoch_{epoch + 1}.pth')
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch + 1
                }, ckpt_path)

    def inference(self, num_steps=999):
        """Run inference on a batch of images"""
        self.model.eval()
        with torch.no_grad():
            # Get a batch of data
            masked_images, real_images, _ = next(iter(self.train_loader))
            masked_images = masked_images.to(self.config['device'])[:2]
            real_images = real_images.to(self.config['device'])[:2]

            for i in range(1, num_steps, 5):
                # Generate samples
                reconstructed = self.sample_diffusion(self.model, torch.randn_like(real_images), i, masked_images,
                                                      self.config['device'])

                # Denormalize images
                masked = self.denormalize(masked_images[:, :3])
                real = self.denormalize(real_images)
                reconstructed = self.denormalize(reconstructed)

                # Create grid and save
                comparison = torch.cat([masked, reconstructed, real], dim=0)
                save_path = os.path.join(self.config['save_dir'], f'inference_samples_{i}.png')
                save_image(comparison, save_path, nrow=masked.size(0))
                print(f"Saved inference samples to {save_path}")