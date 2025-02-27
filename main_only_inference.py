import os

import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from data_handling.celeba_dataset import CelebAMasked
from training.train import Trainer


def main():
    # Configuration
    config = {
        'data_dir': './data/celeba/celeba/img_align_celeba',
        'img_size': 32,
        'mask_size': 4,
        'batch_size': 16,
        'num_workers': 1,
        'lr': 1e-4,
        'epochs': 100,
        'latent_dim': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'mask_type': 'random_rectangle',
        'epoch': 0,
        'timesteps': 1000
    }

    # Initialize dataset and dataloader
    dataset = CelebAMasked(
        root_dir=config['data_dir'],
        img_size=config['img_size'],
        mask_size=config['mask_size'],
        mask_type=config['mask_type']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    # Initialize model
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_dim=config['latent_dim'],
        dim_mults=(1, 2, 4, 8),
        cond_dim=config['latent_dim']
    ).to(config['device'])

    # Load model checkpoint if it exists
    checkpoint_path = f"{config['save_dir']}_strange/model_epoch_latest.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=config['device'])['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        optimizer.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location=config['device'])['optimizer'])
        config['epoch'] = torch.load(checkpoint_path, weights_only=True)['epoch']
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=dataloader,
            config=config,
            optimizer=optimizer
        )
    else:
        raise NotImplementedError("Model checkpoint not found")

    # Run training
    trainer.inference(30)


if __name__ == "__main__":
    main()
