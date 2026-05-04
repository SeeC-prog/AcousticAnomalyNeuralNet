import torch, random, numpy as np
from collections import Counter
from pathlib import Path
from torch import optim

# Local module imports (project structure under ./src or ./)
from data.data_module import MIMIIDatasetModule             # Handles dataset prep, downloading, augs, and dataloaders
from models.transformer import AcousticTransformerEncoder   # Transformer encoder for short term temporal encoding of spectrograms
from models.tcn import TemporalConvNet                      # Temporal conv net for longterm temporal modeling
from models.decoders import DeepDecoder                     # Fully connected decoder for reconstruction of input feaatures from latent space
from models.reconstruction import ReconstructionModel       # Combines encoder, tcn, and decoder into unified model
from trainer.trainer import Trainer                         # Handles training, validation, testing, and scheduling

# Global seed control
def set_seed(seed = 42):
    """
    Ensures full experiment reproducibility across runs.
    Seeds python, numpy, and pytorch rand
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------------ Main entry point ------------------------
# ------------------------------------------------------------------
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42) # Fix random seed reproducibility


    # ------------------------ Data module initialisation ------------------------
    # ----------------------------------------------------------------------------
    data_module = MIMIIDatasetModule(
        machine="fan",                          # Machine Type (options: "fan", "pump", "valve", "slider")
        snrs=["0_dB", "6_dB", "-6_dB"],         # List of Signal to Noise ratios to include
        dataset_names=["seec/mimii-fan-0db"],   # Optional kaggle reference or dataset name
        data_dir="./SoundData",                 # Root directory for cached or downloaded dataset
        sample_rate = 16000,                    # Audio sampling rate in Hz
        n_fft = 1024,                           # FFT window size for spectrogram computation
        hop_length = 256,                       # Hop length for STFI frames - smaller = higher temporal resolution
        n_mels = 128,                           # Number of mel bins in the mel-spectrogram
        batch_size = 64,                        # Mini batch size for training
        target_seconds = 4,                     # length in seconds of each audio segment window
        overlap_train = 0.1,                    # 10% overlap for sliding window in training (data augs)
        overlap_eval = 0.0,                     # No overlap in validation/test for clean validation
    )

    # dataloader creation
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Print dataset composition by anomaly class
    counts = Counter(r["is_abnormal"] for r in data_module.train_dataset.records)
    print("Train level distribution:", counts)


    # ------------------------ Global normalisation stats ------------------------
    # ----------------------------------------------------------------------------
    # Compute mean/std of mel-spectrograms features (using only normal samples to avoid leaks)
    mean, std = data_module.train_dataset.compute_global_stats(use_normal_only = True)
    
    # Save computed statistics for reproducability and reuse
    torch.save({'mean': mean, 'std': std}, 'mel_global_stats.pt')
    print(mean.mean(), mean.std(), std.mean(), std.std())

    # Inject global normalisation parameters into all dataset splits
    for ds in (data_module.train_dataset, data_module.val_dataset, data_module.test_dataset):
        ds.global_mean = mean
        ds.global_std = std


    # ------------------------ Data Split integrity checks ------------------------
    # -----------------------------------------------------------------------------
    def check_split_integrity(dm):
        """
        Ensures that no machine unit (id_00) appears in more than one split
        to prevent data leakage between train/test/val sets
        """
        def groups(ds):
            return set(r["unit_id"] for r in ds.records)
        
        train_g, val_g, test_g = groups(dm.train_dataset), groups(dm.val_dataset), groups(dm.test_dataset)
        print("\n --- Split Integrity Check ---")
        print(f"Train units: {len(train_g)} | Val units: {len(val_g)} | Test units: {len(test_g)}")
        print(f"Overlap Train-Val: {len(train_g & val_g)}")
        print(f"Overlap Train-Test: {len(train_g & test_g)}")
        print(f"Overlap Val-Test: {len(val_g & test_g)}")
        if not ((train_g & val_g) or (train_g & test_g) or (val_g & test_g)):
            print("No unit overlap detected - splits are clean.\n")
        else:
            print("Overlaps found! Check grouping logic.\n")
    check_split_integrity(data_module)
    
    # File level overlap verification (detects duplicated audio across splits)
    def check_file_overlap(ds1, ds2, name1, name2):
        s1 = {r["path"].stem for r in ds1.records}
        s2 = {r["path"].stem for r in ds2.records}
        print(f"Overlap {name1}-{name2}: {len(s1 & s2)} shared files")

    check_file_overlap(data_module.train_dataset, data_module.val_dataset, "train", "val")
    check_file_overlap(data_module.train_dataset, data_module.test_dataset, "train", "test")
    check_file_overlap(data_module.val_dataset, data_module.test_dataset, "val", "test")


    # ------------------------ Model Architection Definition ------------------------
    # -------------------------------------------------------------------------------
    # Encoder: transformer based feature extractor
    encoder = AcousticTransformerEncoder(
        input_dim= 128,                      # Input feature dimension (mel bins)
        embed_dim = 256,                    # Embedding dimension for Transformer
        num_heads = 4,                      # Multi-head self-attention heads
        ff_dim = 512,                       # Feed forward hidden dimension inside transformer block
        num_layers = 2,                     # Number of stacked transformer layers
        dropout = 0.15                      # Dropout rate for regularisation 
    )
    
    # Temporal Convolutional Network: models longer temporal patterns
    tcn = TemporalConvNet(
        input_dim = 256,                        # Input from transformer embedding output
        channel_dims = [256, 128, 64, 32, 8],  # Channels sizes per residual block
        kernel_size=7,                          # Temporal receptive field
        dropout = 0.20                          # Dropout to prevent overfitting
    )
        
    # Decoder: reconstructs mel spectrograms from the low-dimensional bottleneck
    decoder = DeepDecoder(
        input_dim = 8,         # Latent dimensionality after TCN
        hidden_dims = [32, 64], # Hidden layer sizes in MLP decoder
        output_dim = 128,       # Output mel dimensions to match input
        dropout = 0.3
    )
    
    # Construct full model - encoder/tcn/decoder
    model = ReconstructionModel(encoder, tcn, decoder)

    # ------------------------ Optimiser and Scheduler ------------------------
    # -------------------------------------------------------------------------
    num_epochs = 30
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = 1e-5,              # Initial learning rate
        weight_decay = 1e-4     # Regularisation term to reduce overfitting
    )

    # Compute total steps for cosine annealing scheduler 
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs

    # CosineAnnealingLR for smooth learning rate decay over total training duration. 
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = 1e-4,
        pct_start = 0.1,
        div_factor = 10,
        total_steps = total_steps,
    )   


    # ------------------------ Trainer initialisation ------------------------
    # ------------------------------------------------------------------------
    trainer = Trainer(
        model = model,                  # Complete reconstruction model
        optimizer = optimizer,
        train_loader = train_loader,
        val_loader = val_loader,
        scheduler = scheduler,
        device = device,                
    )

    # Training and Evaluation
    trainer.fit(num_epochs)                 # Run training and validation loops
    res = trainer.test(test_loader)         # Evaluate reconstruction metrics on test set
    
    
    # ------------------------ Final reporting ------------------------
    # -----------------------------------------------------------------
    print("\n Final Combined Fan Results")
    print(res)

if __name__ == "__main__":
    main()