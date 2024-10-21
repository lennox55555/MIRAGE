import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
import json
import sounddevice as sd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def load_equations(file_path):
    equations = []
    try:
        with open(file_path, 'r') as f:
            current_equation = []
            for line in f:
                line = line.strip()
                if line == "":  # If it's an empty line, assume new time slice
                    if current_equation:
                        equations.append(current_equation)
                        current_equation = []
                else:
                    # Parse the line to extract magnitude, frequency, and phase
                    parts = line.split(",")
                    magnitude = float(parts[0].split(":")[1].strip())
                    frequency = float(parts[1].split(":")[1].strip())
                    phase = float(parts[2].split(":")[1].strip())
                    current_equation.append({
                        "magnitude": magnitude,
                        "frequency": frequency,
                        "phase": phase
                    })

            # Append the last equation if it's not empty
            if current_equation:
                equations.append(current_equation)

        return equations
    except Exception as e:
        raise RuntimeError(f"Error loading equations file: {str(e)}")

def synthesize_audio_from_equations(equations, sample_rate, hop_length):
    n_fft = config.WINDOW_SIZE
    n_frames = len(equations)
    Zxx = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)

    for i, eq_components in enumerate(equations):
        for component in eq_components:
            freq_index = int(component["frequency"] * n_fft / sample_rate)
            if freq_index < Zxx.shape[0]:
                Zxx[freq_index, i] = component["magnitude"] * np.exp(1j * component["phase"])

    return librosa.istft(Zxx, hop_length=hop_length, window='hann')

def normalize_audio(data):
    return data / np.max(np.abs(data))

def play_audio(audio, sample_rate):
    sd.play(audio, sample_rate)
    sd.wait()

def save_wav(audio, sample_rate, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sf.write(file_path, audio, sample_rate, subtype='PCM_24')
        print(f"Saved synthesized audio to: {file_path}")
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")

def plot_audio(audio, sample_rate, title):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sample_rate, title):
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def main():
    try:
        if config.INPUT_FORMULA is None:
            raise ValueError("INPUT_FORMULA not set. Run song_to_formula.py first.")

        print(f"Input formula file: {config.INPUT_FORMULA}")
        print(f"Output directory: {config.OUTPUT_DIR}")

        equations = load_equations(config.INPUT_FORMULA)
        print(f"Loaded {len(equations)} equations.")

        sample_rate = config.TARGET_SR
        synthesized_audio = synthesize_audio_from_equations(equations, sample_rate, config.HOP_LENGTH)
        synthesized_audio = normalize_audio(synthesized_audio)

        output_dir = getattr(config, 'OUTPUT_DIR', os.path.dirname(config.INPUT_FORMULA))
        base_name = os.path.splitext(os.path.basename(config.INPUT_FORMULA))[0]
        synth_audio_path = os.path.join(output_dir, f"{base_name}_synthesized.wav")
        save_wav(synthesized_audio, sample_rate, synth_audio_path)

        print("\nPlaying synthesized audio...")
        play_audio(synthesized_audio, sample_rate)

        plot_audio(synthesized_audio, sample_rate, 'Synthesized Audio')
        plot_spectrogram(synthesized_audio, sample_rate, 'Spectrogram of Synthesized Audio')

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()