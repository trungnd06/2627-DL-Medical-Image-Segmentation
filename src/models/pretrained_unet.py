"""M2/A1: SMP U-Net ResNet34; decoder initialization shared across variants."""
import hashlib
import torch
from torch import nn


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PretrainedUNet(nn.Module):
    def __init__(self, encoder_weights='imagenet', decoder_channels=(256, 128, 64, 32, 16), init_seed=42):
        super().__init__()
        import segmentation_models_pytorch as smp
        from torchvision.models import ResNet34_Weights
        if encoder_weights not in ('imagenet', None):
            raise ValueError('encoder_weights must be imagenet or None')
        # Construct both variants identically, then replace only encoder weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.network = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                encoder_depth=5, decoder_channels=tuple(decoder_channels),
                decoder_use_norm='batchnorm', decoder_interpolation='nearest',
                in_channels=3, classes=1, activation=None)
        self.provenance = {'library': 'segmentation_models_pytorch', 'version': smp.__version__,
            'encoder': 'resnet34', 'encoder_weights': encoder_weights,
            'decoder_channels': list(decoder_channels), 'decoder_normalization': 'batchnorm',
            'decoder_interpolation': 'nearest', 'init_seed': init_seed,
            'decoder_initial_sha256': state_digest(self.network.decoder.state_dict()),
            'head_initial_sha256': state_digest(self.network.segmentation_head.state_dict()),
            'weights_url': None}
        if encoder_weights is not None:
            weights = ResNet34_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=True, check_hash=True)
            # SMP's ResNet encoder has no classification head.
            state = {k: v for k, v in state.items() if not k.startswith('fc.')}
            self.network.encoder.load_state_dict(state, strict=True)
            self.provenance['weights_url'] = weights.url
        self.provenance['encoder_initial_sha256'] = state_digest(self.network.encoder.state_dict())
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, x):
        return self.network(x)
