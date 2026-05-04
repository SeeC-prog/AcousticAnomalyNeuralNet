
import torch.nn as nn


# -------------------------------
# Deep Decoder
#
# Purpose:
#   - More capable reconstruction head combining temporal 
#     MLP layers and convolutional refinement.
#   - Uses resideral skip connection from input to ouput and 
#     layernorm for stabalised reconstruction
#
# Expected I/O Shapes:
#   Input: [B, T, input_dim]
#   Output: [B, T, output_dim]
# -------------------------------
class DeepDecoder(nn.Module):
    def __init__(self, input_dim = 16, hidden_dims = [64, 128], output_dim = 128, dropout = 0.2):
        """
        Args:
            input_dim: Dimensionality of latent features
            hidden_dims: List of hidden layer sizes for tempral MLP
            output_dim: Final mel feature dimentionality
            dropout: Dropout probability for regularisation
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Temporal fully connected stack (per time step)
        self.temporal = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Convolutional refinement (1D conv across time)
        self.conv_refine = nn.Sequential(
            nn.Conv1d(hidden_dims[1], hidden_dims[1], kernel_size = 5, padding = 2),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Conv1d(hidden_dims[1], output_dim, kernel_size = 3, padding = 1),
        )

        # Final normalisation and residual shortcut
        self.final_norm = nn.LayerNorm(output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim)
        
        
    def forward(self, x):
        """
        Forward pass through decoder
        Combines temporal MLP, conv refinement, and residual connections
        
        Args:
            x: Latent representation [B, T, input_dim]
            
        Returns:
            Reconstructed mel features [B, T, output_dim]
        """
        
        # Temporal projection (frame wise)
        out = self.temporal(x)
        
        # Temporal convolution refinement (time axis)
        out = out.transpose(1, 2)       # [B, hidden_dim, T]
        out = self.conv_refine(out)     # [B, output_dim, T]
        out = out.transpose(1, 2)       # [B, T, output_dim]

        # Residual conection from input 
        res = self.shortcut(x)          # [B, T, output_dim]
        
        # Combine and normalise
        out = out + res
        out = self.final_norm(out)
        return out