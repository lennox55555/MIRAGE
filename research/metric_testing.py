import librosa
import numpy as np
import matplotlib.pyplot as plt
import config


# Function to load audio and resample if necessary
def load_audio(filename, target_sr=16000):
    y, sr = librosa.load(filename, sr=target_sr)
    return y, sr


# Function to compute percentage difference between original and perturbed signals
def calculate_difference_percentage(original, perturbed):
    # Ensure both signals are of the same length
    min_length = min(len(original), len(perturbed))
    original = original[:min_length]
    perturbed = perturbed[:min_length]

    # Calculate the absolute difference
    diff = np.abs(original - perturbed)

    # Calculate the percentage difference based on amplitude
    percentage_diff = (np.sum(diff) / np.sum(np.abs(original))) * 100
    return percentage_diff


# Function to visualize the original and perturbed waveforms
def visualize_waveforms(original, perturbed, sr):
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.title("Original Audio")
    librosa.display.waveshow(original, sr=sr)

    plt.subplot(2, 1, 2)
    plt.title("Perturbed Audio")
    librosa.display.waveshow(perturbed, sr=sr)

    plt.tight_layout()
    plt.show()


# Function to compare and visualize differences
def compare_audio(original_file, perturbed_file):
    # Load both audio files
    original, sr1 = load_audio(original_file)
    perturbed, sr2 = load_audio(perturbed_file)

    # Ensure both audios have the same sample rate
    assert sr1 == sr2, "Sample rates do not match!"

    # Calculate the difference percentage
    diff_percentage = calculate_difference_percentage(original, perturbed)
    print(f"Difference Percentage: {diff_percentage:.2f}%")

    # Visualize both waveforms
    visualize_waveforms(original, perturbed, sr1)


# Main function
def main():
    original_wav = config.INPUT_SONG  # Input song from config.py
    perturbed_wav = config.PERTURB_SONG  # Perturbed song from config.py
    compare_audio(original_wav, perturbed_wav)


# If the script is run directly, execute the main function
if __name__ == "__main__":
    main()
