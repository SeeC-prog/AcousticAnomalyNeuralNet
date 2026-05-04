import torch.nn as nn
import torch

# --------------------------------------------------------
# ReconstructionModel
#
# Purpose:
#   - Reconstructs input spectrogram segments (Tx n_mels)
#     from compressed latent embeddings using an encoder-TCN-decoder pipeline
#   - Used in unsupervised acoustic anomaly detection where
#     reconstruction error (MSE) serves as the anomaly score
#
# Expected I/O shapes:
#   Input: [B, T, n_mels] Mel spectrogram frames
#   Output: [B, T, n_mels] Reconstructed Mel spectrogram
# --------------------------------------------------------

class ReconstructionModel(nn.Module):
    def __init__(self, encoder, tcn, decoder):
        """
        Args:
            - encoder: Transformer encoder feature extractor 
            - tcn: Temporal Conv Net
            - decoder: reconstruction head
        """
        super().__init__()
        self.encoder = encoder
        self.tcn = tcn
        self.decoder = decoder
        
        latent_dim = getattr(decoder, "input_dim", None)
        if latent_dim is None:
            raise ValueError("Decoder must define input_dim for gating layer")
            
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through the reconstruction model.
        
        Args:
            x: input tensor [B, T, n_mels]
        """
        # Encode short-term spectral context
        z = self.encoder(x)     # [B, T, d_model]
        
        # Model long-term tempral dependencies
        z = self.tcn(z)         # [B, T, d_model]

        g = self.gate(z)
        z = z * g
        
        if self.training:
            z = z + 0.01 * torch.randn_like(z)
        
        # Decode latent to reconstruct mel spectrogram
        recon = self.decoder(z) # [B, T, n_mels]
        return recon