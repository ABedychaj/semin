import math

from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, scheduler_type='linear', num_warmup_steps=0, num_training_steps=1000):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if scheduler_type == 'linear':
            return max(0.0, float(num_training_steps - current_step) /
                       float(max(1, num_training_steps - num_warmup_steps)))
        elif scheduler_type == 'cosine':
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi *
                                                  float(current_step - num_warmup_steps) /
                                                  float(max(1, num_training_steps - num_warmup_steps)))))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)
