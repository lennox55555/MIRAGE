import os

# Get the directory of this config file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Input file path for song_to_formula.py and stem_separation.py
INPUT_SONG = "/Users/lennox/Documents/machineLearning/research/musicMask/Westy - KING OF THE NIGHT.wav"

# Perturbed song file path (new)
PERTURB_SONG = "/Users/lennox/Documents/machineLearning/research/musicMask/Westy - KING OF THE NIGHT.wav"

# Output directory (now using an absolute path)
OUTPUT_DIR = os.path.join(CONFIG_DIR, "data", "perturbed_wav")

# STFT parameters
WINDOW_SIZE = 8192
HOP_LENGTH = 2048

# Target sample rate
TARGET_SR = 44100

# Input file path for formula_to_song.py (this will be the output of song_to_formula.py)
INPUT_FORMULA = os.path.join(CONFIG_DIR, "data", "perturbed_wav", "Westy - KING OF THE NIGHT_equations.txt")
