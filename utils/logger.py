import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

        # Create text log file
        self.text_log_path = os.path.join(log_dir, 'training_log.txt')
        self._init_text_log()

    def _init_text_log(self):
        with open(self.text_log_path, 'w') as f:
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

    def log(self, data_dict, step=None):
        """Log data to both tensorboard and text file"""
        # Log to tensorboard
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

        # Log to text file
        log_entry = f"Step {step}: " if step is not None else ""
        log_entry += ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in data_dict.items()])

        with open(self.text_log_path, 'a') as f:
            f.write(log_entry + "\n")

    def log_images(self, tag, images, step):
        """Log images to tensorboard"""
        self.writer.add_images(tag, images, step)

    def close(self):
        self.writer.close()


# Example usage:
if __name__ == "__main__":
    logger = Logger('test_logs')
    logger.log({'loss': 0.5, 'lr': 1e-4}, step=0)
    logger.close()