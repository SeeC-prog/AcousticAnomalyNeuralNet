

import torch
import torch.nn as nn
import math

# --------------------------------------------
# AcousticTransformerEncoder
#
# Purpose:
#   - Extracts contextualised embeddings from mel-spectrograms
#     using a transformer encoder stack
#   - Learns both local and global acoustic relationships
#
# Components:
#   - PositionalEncoding: Adds sinusoidal position information
#   - Linear input projection: maps input_dim to embed_dim
#   - Transformer encoder layers with GELU activiations
#   - Optional normalisation and dropout for stability
#
# Expectped I/O:
#   Input: [B, T, input_dim] mel features over time
#   Output: [B, T, embed_dim] contextualised embeddings
# --------------------------------------------


class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encoding as 
    described in "Attention is all you need" by Vaswani et al., 2017
    
    Inject deterministic position-dependent signals into embeddings to 
    allow the model to capture order without recurrences
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds positional encoding to input embeddings
        
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :]


# Transformer encoder
class AcousticTransformerEncoder(nn.Module):
    """
    Transformer based encoder for acoustic feature sequences
    
    Args:
        input_dim: Input feature dimension (n_mels)
        embed_dim: Embedding dimension for transformer
        num_heads: Number of attention heads
        ff_dim: Feedforward hidden dimension
        num_layers: Number of transformer encoder layers
        dropout: Probability of dropout
        max_seq_len: Max sequence length for positional encoding. 
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()

        # Linear projection from input mel features to transformer space
        self.input_proj = nn.Linear(input_dim, embed_dim)

        #Positional encoding for temporal order awareness
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        #Transformer encoder stack (multi-head self-attention with FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = ff_dim,
            dropout = dropout,
            activation = "gelu", 
            batch_first = True, #Use to ensure input format stays [B, T, D]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final normalisation and dropout for stability
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_dropout = nn.Dropout(0.2)
        self.input_dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass through the acoustic transformer encoder
        
        Args:
            x: input tensor [B, T, input_dim] mel spectrogram over time
        Returns:
            Contextualised embeddings [B, T, embed_dim]
        """
        # Project input to embedding dimension
        x = self.input_proj(x)
        x = self.input_dropout(x)
        
        # add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer layers
        x = self.transformer(x)
        
        # Normalise and apply final dropout
        x = self.final_norm(x)
        x = self.final_dropout(x)
        return x