import torch

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        from collections import deque
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count != 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1] if self.deque else None

    def __str__(self):
        return self.fmt.format(median=self.median, global_avg=self.global_avg, max=self.max, value=self.value)