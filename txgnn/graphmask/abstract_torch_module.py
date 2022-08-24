import torch
import sys


class AbstractTorchModule(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    def save(self, path):
        print("Saving to path " + path, file=sys.stderr)
        torch.save(self.state_dict(), path)

    def load(self, path):
        print("Loading from path " + path, file=sys.stderr)

        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_device(self, device):
        self.device = device
        self.to(device)
