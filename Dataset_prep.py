import os
import math
import glob
import random
import torch
import torchaudio
import logging
from pathlib import Path
from tqdm import tqdm
from torchaudio.transforms import Resample

# ==============================================================================
# --- 1. CONFIGURATION: UPDATE THESE PATHS ---
# ==============================================================================

# The full path to your root 'Clean_Dataset' folder.
CLEAN_DATA_ROOT = r"Clean_Dataset"

# The full path to your root 'noisy_dataset' folder.
NOISE_DATA_ROOT = r"E:\noisy_dataset"

# The path where the new 'generated_dataset' folder will be created.
OUTPUT_ROOT = "output_noise_dataset"

# --- This list defines EXACTLY which noise types will be processed. ---
NOISE_TYPES_TO_PROCESS = {
    "Animal": ["dog", "chirping_birds", "cat"],
    "objects": ["air_conditioner", "keyboard_typing", "vacuum_cleaner", "washing_machine"],
    "Nature": ["wind", "thunderstorm", "sea_waves", "water_drops"],
    "Environment": ["car_horn", "siren", "street_music", "STRAFFIC_16k", "PCAFETER_16k"]
}

# ==============================================================================
# --- 2. PROCESSING PARAMETERS (Generally, no changes needed here) ---
# ==============================================================================

TARGET_SAMPLE_RATE = 16000
SNR_LEVELS_DB = [-20, -15, -10, -5, 5, 10, 15, 20]


# ==============================================================================
# --- 3. HELPER FUNCTIONS ---
# ==============================================================================

def load_audio(file_path: Path, target_sr: int, logger) -> torch.Tensor:
    """Loads, resamples, and converts an audio file to a mono tensor."""
    try:
        waveform, original_sr = torchaudio.load(file_path)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None

    if original_sr != target_sr:
        resampler = Resample(original_sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform

def save_audio(waveform: torch.Tensor, file_path: Path, sample_rate: int):
    """Saves a waveform to a .wav file, creating directories if needed."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(file_path, waveform, sample_rate)

def mix_at_snr(clean, noise, snr, eps=1e-10):
    """
    Mixes clean and noise at a target SNR.
    NOTE: This optimized version assumes clean and noise tensors have the same length.
    """
    power_clean = torch.mean(clean.pow(2))
    power_noise = torch.mean(noise.pow(2))

    # Calculate the scaling factor for the noise to achieve the target SNR
    scale_factor = torch.sqrt((power_clean / (power_noise + eps)) * (10 ** (-snr / 10)))

    scaled_noise = noise * scale_factor
    mixture = clean + scaled_noise

    # Prevent clipping by normalizing if the max amplitude exceeds 1.0
    max_amplitude = mixture.abs().max()
    if max_amplitude > 1.0:
        mixture = mixture / max_amplitude

    return mixture


# ==============================================================================
# --- 4. MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == "__main__":
    # --- Set up proper logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    logger.info("--- Starting Noisy Dataset Generation (Optimized for equal-length audio) ---")
    
    clean_root = Path(CLEAN_DATA_ROOT)
    noise_root = Path(NOISE_DATA_ROOT)
    output_root = Path(OUTPUT_ROOT)
    
    # --- PHASE 1: PREPARATION ---
    logger.info("Phase 1: Scanning directories and preparing noise pools...")

    clean_file_paths = sorted(list(clean_root.rglob("*.wav")))
    
    shuffled_noise_pools = {}
    total_noise_files = 0
    for category, types in NOISE_TYPES_TO_PROCESS.items():
        for type_name in types:
            noise_dir = noise_root / category / type_name
            if not noise_dir.is_dir():
                logger.warning(f"Noise directory not found, skipping: {noise_dir}")
                continue
            
            noise_paths = list(noise_dir.rglob("*.wav"))
            if not noise_paths:
                logger.warning(f"No .wav files found in {noise_dir}, skipping.")
                continue

            random.shuffle(noise_paths)
            pool_key = f"{category}/{type_name}"
            shuffled_noise_pools[pool_key] = noise_paths
            total_noise_files += len(noise_paths)

    if not clean_file_paths or not shuffled_noise_pools:
        logger.error("No clean files or valid noise files found. Please check configuration.")
        exit()

    logger.info(f"Found {len(clean_file_paths)} clean audio files.")
    logger.info(f"Found {total_noise_files} noise files across {len(shuffled_noise_pools)} selected types.")
    total_mixtures = len(clean_file_paths) * len(shuffled_noise_pools) * len(SNR_LEVELS_DB)
    logger.info(f"Total mixtures to generate: {total_mixtures:,}")
    
    # --- PHASE 2: MAIN PROCESSING LOOP ---
    logger.info("Phase 2: Generating mixtures... (Progress bar will appear below)")

    for i, clean_path in enumerate(tqdm(clean_file_paths, desc="Processing Clean Files")):
        
        clean_waveform = load_audio(clean_path, TARGET_SAMPLE_RATE, logger)
        if clean_waveform is None:
            continue

        for pool_key, noise_list in shuffled_noise_pools.items():
            
            noise_index = i % len(noise_list)
            selected_noise_path = noise_list[noise_index]

            noise_waveform = load_audio(selected_noise_path, TARGET_SAMPLE_RATE, logger)
            if noise_waveform is None:
                continue
            
            # Final check to ensure durations match, just in case.
            if clean_waveform.shape[1] != noise_waveform.shape[1]:
                logger.warning(f"Duration mismatch! Skipping mix for {clean_path.name} and {selected_noise_path.name}")
                continue

            category, type_name = pool_key.split('/')
            gender = clean_path.parent.parent.name
            speaker_id = clean_path.parent.name
            clean_filename_base = clean_path.stem

            for snr in SNR_LEVELS_DB:
                
                mixed_waveform = mix_at_snr(clean_waveform, noise_waveform, snr)

                snr_label = f"SNR_{snr}" if snr <= 0 else f"SNR_+{snr}"
                output_dir = output_root / category / type_name / gender / speaker_id / snr_label
                
                output_filename = f"{clean_filename_base}_{snr}dB.wav"
                output_filepath = output_dir / output_filename

                save_audio(mixed_waveform, output_filepath, TARGET_SAMPLE_RATE)

    logger.info("--- Dataset Generation Complete! ---")