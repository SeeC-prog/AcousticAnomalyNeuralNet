import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# -----------------------------------------------------
# Temporal Convolutional Network
#
# Purpose:
#   - Models long-range temporal dependencies in latent sequences
#      using dilated 1D convolutions with a residual connection.
#   - Designed to follow tranformer encoders for temporal contect modeling
#
# Components:
#   - Chomp1d:          Removes causal padding to keep output length consistent
#   - TemporalBlock     Residual block with dilated causal convolutions
#   - TemporalConvNet   Stack of TemporalBlocks with exponentially increasing dialation
#
# Expected I/O:
#   Input: [B, T, D_in]
#   Output: [B, T, D_out]
# -----------------------------------------------------


class Chomp1d(nn.Module):
    """
    Removes extra padding after convolution to preserve causality. Like pacman apparently.
    Ensures each output at time t only depends on inputs < or = to t. 
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

# TemporalBlock - building blocks
class TemporalBlock(nn.Module):
    """
    Single residual block in Temporal Conv Net (TCN)
    
    Structure:
        Conv1d - chomp - batchnorm - GELU - Dropout
        Conv1d - chome - batchnorm - GELU - Dropout
        
    Args:
        in_channels: Input feature channels
        out_channels: output feature channels
        kernel_size: Size of temporal convultional kernel_size
        dialtion: dialtion factor (controls receptive field)
        padding: amount of padding to preserve sequence length
        dropout: dropout probability for regularisation
    """
    
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
        super().__init__()

        # First causal convolution block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        
        # Weight initialisation using kaiming - ideally for stability
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
        nn.init.constant_(self.conv2.bias, 0.0)
        

        # Combine both blocks into a single sequential system
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1, 
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2   
        )
        
        # Residual projetion (if input/output dims differ)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) 
            if in_channels != out_channels else nn.Identity()
        )

        self.final_gelu = nn.GELU()
        

        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C_in, T]
        Returns:
            Output tensor [B, C_out, T]
        """
        out = self.net(x)
        res = self.downsample(x)
        return self.final_gelu(out + res) #add residual


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation
        
    Args:
        input_dim: input feature dimension from encoders
        channel_dims: list of ouput channels for each layers
        kernel_size: Temporal convolution kernal size
        dropout: dropout probability
            
    Example:
        TemporalConvNet(
            input_dim = 128,
            channel_dims = [128, 128, 128],
            kernal_size = 3,
            dropout = 0.2,
        )
    """
    def __init__(self, input_dim, channel_dims, kernel_size, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(channel_dims)
        
        # Build stacked temporal blocks
        for i in range(num_levels):
            dilation = 2 ** i       # doubles the receptive field each layer
            in_channels = input_dim if i == 0 else channel_dims[i-1]
            out_channels = channel_dims[i]
            padding = (kernel_size - 1) * dilation
            
            layers.append(
                TemporalBlock(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    dilation, 
                    padding, 
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(channel_dims[-1])

    def forward(self, x):
        """
        Forward pass through the TCN stacked
        Converts input from [B, T, D] to [B, D, T] for Conv1d operations
        
        Args:
            x: input sequence [B, T, D]
        Returns:
            Ouput sequence [B, T, D_out]
        """
        x = x.transpose(1, 2) #(B, D, T)
        out = self.network(x).transpose(1, 2) #(B, D_out, T) to [B, T, D_out]
        out = self.final_norm(out)
        return out