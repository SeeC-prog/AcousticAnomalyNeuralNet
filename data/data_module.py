
import re, random, zipfile, subprocess
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from .dataset import MIMIIDataset
from typing import List, Dict
# ------------------------------------
# MIMIIDatasetModule
# Purpose:
#   - Stores all data handling for the MIMII acoustic anomaly dataset
#   - Automates discovery, extraction, and loading of .wav files for a
#     given machine type (fan, pump, slider, valve) and each SNR subset
#   - Handles grouping and splitting by machine unit ID's to prevent data leakage
#     between training, validaion, and testing sets.
#   - Returns pytorch dataset and dataloader objects ready to use
# 
# Data Flow:
#   - Zip - Extract / Download (if needed)
#         - Parse file paths (machine, unit_id, label)
#         - Build record dictionaries
#         - Group by unit -> train/val/test split
#         - Create MIMIIDataset objects with mel preprocessing
#         - Wrap in dataloaders for model training
# ------------------------------------

class MIMIIDatasetModule:
    """ Main dataset management interface for MIMII audio anomaly experments"""
    _rx = re.compile(r"/(fan|pump|slider|valve)/(id_\d{2})/(normal|abnormal)/", re.IGNORECASE)

    def __init__(
        self,
        machine:str = "fan",
        snrs: List[str] = ["0_dB"],
        dataset_names: List[str] = None,
        data_dir: str = "./SoundData",
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        val_split: float = 0.2,
        test_split: float = 0.1,
        batch_size: int = 32,
        seed: int = 42,
        target_seconds: float = 2.0,
        overlap_train:float = 0.5,
        overlap_eval: float = 0.0,
    ):
        """
        The module is initiliased once per machine type (fan/pump/slider/valve)
        It preps all datasets and dataloaders end to end
        
        Args:
            - Machine: which machine type to load (see above)
            - snrs: list of SNR levels (0_dB, -6_dB, 6_dB)
            - dataset_names: optional kaggle dataset names for download
            - data_dir: directory where dataset is located
            - sample_rate: sampling frequency of audio signals
            - n-fft, hop_length, n_mels: Mel spectrogram parameters
            - val_split, test_split,: Ratios for validation/test splits
            - batch_size: number of samples per batch for loaders
            - seed: random seed for deterministic splits
            - target seconds: duration of each spectrogram segment
            - overlap_train/val: Sliding window overlap ratios
        """
        
        # --------------- Core configuration ---------------
        self.dataset_names = dataset_names
        self.machine = machine
        self.snrs = snrs
        self.data_dir = (Path.cwd() / data_dir).resolve() # Convert to absolute parth
        self.data_dir.mkdir(parents=True, exist_ok=True)  # Create folder if missing

        # Audio preprocessing configuration
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_seconds = target_seconds

        # Data splitting parameters
        self.val_split = val_split 
        self.test_split = test_split
        self.batch_size = batch_size
        
        #Reproducibility
        random.seed(seed)
        
        # Window overlap ratios (train/eval)
        self.overlap_train = overlap_train
        self.overlap_eval = overlap_eval

       
        # Collect .wav file paths across all request SNR subsets
        wavs = []
        for snr in snrs:
            wavs.extend(self._collect_subset(snr))  # Gather all wav paths for each SNR
        
        # Build metadata records for all .wav files
        records_all = self._build_records(wavs)
        
        # Group and split by unit_id into train, validation and test
        ds_train_all, ds_val, ds_test = self._split_grouped(records_all)

        # Filter training records to remove all abnormal samples (reconstruction)
        train_recs = [r for r in ds_train_all.records if r["is_abnormal"] == 0]
        print(f"[Recon Mode] Train filtered to normals -> {len(train_recs)} records")

        # Build datasets
        self.train_dataset = MIMIIDataset(
            train_recs, 
            self.sample_rate, 
            self.n_fft, self.hop_length, 
            self.n_mels, train= True, 
            target_seconds = self.target_seconds, 
            overlap = self.overlap_train
        )
        self.val_dataset = ds_val
        self.test_dataset = ds_test


    # ----------------------------- helpers -----------------------------
    # _collect_subset: locate or download all .wav files foe a given SNR
    def _collect_subset(self, snr: str) -> List[Path]:
        """
        Locates and extracts all .wav files for a specific SNR (0_dB_fan)
        Search priority:
            - 1. Use existing extraced folder if already available
            - 2. Extract from a local zip if found
            - 3. Download from kaggle if dataset names provided
        """
        machine = self.machine
        subset = self.data_dir / f"{snr}_{machine}" #/SoundData/0_dB_fan
        root = subset / machine                     #/SoundData/0_dB_fan/fan

        # Already extracted folder
        if root.exists() and any(root.rglob("*.wav")):
            return sorted(root.rglob("*.wav"))

        # Try local zip file
        zname = f"{snr}_{machine}.zip"
        local = next((p for p in self.data_dir.glob("*") if p.name.lower() == zname.lower()), None)
        if local:
            subset.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local, "r") as zf:
                zf.extractall(subset)
            return sorted(root.rglob("*.wav"))

        # Try Kaggle Download
        if self.dataset_names:
            print(f"[Kaggle] fetching {machine} @ {snr} ... ")
            match = [d for d in self.dataset_names
                     if machine in d.lower() and snr.replace("_", "").lower() in d.lower()]
            if not match:
                raise FileNotFoundError(f"No dataset entry for {machine} @ {snr}")
            
            # Download dataset from Kaggle CLI
            subprocess.run(["kaggle", "datasets", "download", "-d", match[0], "-p", str(self.data_dir)], check=True)
            
            # Extract zip file after download
            zip_path = next((p for p in self.data_dir.glob(f"*{snr}*{machine}*.zip")), None)
            if not zip_path:
                raise FileNotFoundError(f"Downloaded but no zip for {machine}")
            
            subset.mkdir(parents=True, exist_ok = True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(subset)
                
            return sorted(root.rglob("*.wav"))
        
        # No valid source found
        raise FileNotFoundError(f"No local or downloadable zip file found for {machine} @ {snr}")


    # _parse: extract machine unit and label info from path
    def _parse(self, p: Path):
        """
        Parses a file path into:
            - unit_id (id_00)
            - label (0 for normal, 1 for abnormal)
        """
        s = str(p).replace("\\","/").lower() # Normalise for windows paths
        m = self._rx.search(s)
        if not m:
            raise ValueError(f"Bad path layout: {p}")
        _machine, unit_id, lab = m.group(1), m.group(2), m.group(3)
        return unit_id, 1 if lab == "abnormal" else 0
        
        
    # _build_records: Constructs full metadata list for all audio files    
    def _build_records(self, wavs: List[Path]) -> List[Dict]:
        """
        Builds metadata disctionary for each file.
        Adds consistent indices and SNR tags for later grouping
        """
        units = set()
        parsed = []
        
        # Parse each file to extract unit ID and abnormal flag
        for p in wavs:
            unit_id, is_abnormal = self._parse(p)
            parsed.append((p, unit_id, is_abnormal))
            units.add(unit_id)
        
        # Assign each unit a global integer index
        unit_global = {uid: i for i, uid in enumerate(sorted(units))}

        # build final records dicts (path, snr, labels, etc)
        recs=[]
        for p, uid, ab in parsed:
            parts = str(p).replace("\\", "/").split("/")
            # Extract SNR info (folder like 0_dB)
            if len(parts) >= 5 and "_db" in parts[-5].lower():
                snr = parts[-5]
            else:
                snr = next((x for x in parts if "_db" in x.lower()), "unknown_snr")

            recs.append({
                "path": p,
                "unit_id": uid,
                "snr": snr,
                "is_abnormal": ab,
                "unit_global_idx": unit_global[uid],
            })
        print(f"[build records] Parsed {len(recs)} files across {len(unit_global)} units.")
        print(f"Example: {recs[0]['path']} -> SNR = {recs[0]['snr']}")
        return recs
        
    # _split_grouped: Divide records into train/val/test groups by unit_id    
    def _split_grouped(self, recs: List[Dict]):
        """
        Splits dataset by unit_id
        Ensures that sounds from the same machine unit do not appear in multiple splits.
        This enforces generalisation across unseen units.
        """
        # Group all record indices by unit_id
        per_unit = defaultdict(list)
        for i, r in enumerate(recs):
            group_key = f"{r['unit_id']}"
            per_unit[group_key].append(i)

        units = list(per_unit.keys())

        # If machine is FAN - fixed split for generalisation testing
        if self.machine == "fan":
            k = len(units)
            print("[FAN] using fixed split for units")
            u_train = {"id_00", "id_06"}
            u_val = {"id_04"}
            u_test = {"id_02"}
            
        else:
            # Shuffle unit order for random splitting
            random.shuffle(units)
            k = len(units)
            
            # Special handling for small datasets with less than 3 units (ie one fan)
            if k < 3:
                print(f"[{self.machine}] Only {k} units(s): {units} -> Using per=file non-overlapping split")
                
                # Collect unique file stems to avoid duplicates
                file_stems = sorted({r["path"].stem for r in recs})
                random.shuffle(file_stems)

                # Compute counts for each split
                n_total = len(file_stems)
                n_test = max(1, int(self.test_split * n_total))
                n_val = max(1, int(self.val_split * (n_total - n_test)))
                n_train = n_total - n_val - n_test

                # Partition file stems into sets
                train_files = set(file_stems[:n_train])
                val_files = set(file_stems[n_train:n_train + n_val])
                test_files = set(file_stems[n_train + n_val:])

                # Helper to collect full record objects
                def collect_small(split_files):
                    return [r for r in recs if r["path"].stem in split_files]

                # Build dataset objects for this case
                train_recs = collect_small(train_files)
                val_recs = collect_small(val_files)
                test_recs = collect_small(test_files)

                print(f"[Smaples] train = {len(train_recs)} val = {len(val_recs)} test = {len(test_recs)}")

                ds_train = MIMIIDataset(train_recs, self.sample_rate, self.n_fft, self.hop_length, self.n_mels, train=True, target_seconds = self.target_seconds, overlap=self.overlap_train)
                ds_val   = MIMIIDataset(val_recs,   self.sample_rate, self.n_fft, self.hop_length, self.n_mels, train=False, target_seconds = self.target_seconds, overlap=self.overlap_eval)
                ds_test  = MIMIIDataset(test_recs,  self.sample_rate, self.n_fft, self.hop_length, self.n_mels, train=False, target_seconds = self.target_seconds, overlap=self.overlap_eval)
                return ds_train, ds_val, ds_test
                
                
            # Normal case: Unit-level random split 
            n_test = max(1, round(self.test_split * k))
            n_val = max(1, round(self.val_split * (k - n_test)))
            n_train = k - n_val - n_test

            u_train = set(units[:n_train])
            u_val = set(units[n_train:n_train + n_val])
            u_test = set(units[n_train + n_val:])

        # Print diagnostic summary of split
        print(f"[{self.machine}] units = {k} train = {len(u_train)} val={len(u_val)} test={len(u_test)}")
        print(f"train={sorted(u_train)} val={sorted(u_val)} test = {sorted(u_test)}")


        # Helper: collect all records belonging to a group of unit_ids
        def collect(group):
            idxs = []
            for uid in group:
                idxs.extend(per_unit[uid])
            return [recs[i] for i in sorted(idxs)]

        train_recs = collect(u_train)
        val_recs   = collect(u_val)
        test_recs  = collect(u_test)

        print(f"[Samples] train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

        # Construct dataset objects
        ds_train = MIMIIDataset(train_recs, self.sample_rate, self.n_fft, self.hop_length, self.n_mels, train=True, target_seconds = self.target_seconds, overlap=self.overlap_train)
        ds_val   = MIMIIDataset(val_recs,   self.sample_rate, self.n_fft, self.hop_length, self.n_mels, train=False, target_seconds = self.target_seconds, overlap=self.overlap_eval)
        ds_test  = MIMIIDataset(test_recs,  self.sample_rate, self.n_fft, self.hop_length, self.n_mels, train=False, target_seconds = self.target_seconds, overlap=self.overlap_eval)
        return ds_train, ds_val, ds_test


    # Dataloader Accessors
    def train_dataloader(self):
        """ Return shuffled DataLoader for training set"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        """ Return sequential DataLoader for validation set"""
        return DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        """Return sequential DataLoader for test set"""
        return DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False)