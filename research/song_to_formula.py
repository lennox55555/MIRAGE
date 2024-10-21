import numpy as np
import librosa
import soundfile as sf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_wav(file_path):
    try:
        data, sample_rate = sf.read(file_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Convert stereo to mono
        return sample_rate, data
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {str(e)}")


def normalize_audio(data):
    return data / np.max(np.abs(data))


def perform_stft(data, sample_rate, window_size, hop_length):
    return librosa.stft(data, n_fft=window_size, hop_length=hop_length, window='hann')


def create_equation(frequencies, Zxx_slice):
    magnitudes = np.abs(Zxx_slice)
    phases = np.angle(Zxx_slice)

    equation_components = []
    for i in range(len(frequencies)):
        if magnitudes[i] > 1e-8:  # Ignore very small magnitudes
            equation_components.append({
                "magnitude": float(magnitudes[i]),
                "frequency": float(frequencies[i]),
                "phase": float(phases[i])
            })

    return equation_components


def save_equations_to_text(equations, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for equation in equations:
                for component in equation:
                    f.write(f"Magnitude: {component['magnitude']}, Frequency: {component['frequency']}, Phase: {component['phase']}\n")
                f.write("\n")  # Separate each time slice
        print(f"Saved equations to: {file_path}")
    except Exception as e:
        print(f"Error saving equations: {str(e)}")


def main():
    try:
        print(f"Input file: {config.INPUT_SONG}")
        print(f"Output directory: {config.OUTPUT_DIR}")

        sample_rate, data = load_wav(config.INPUT_SONG)
        data = normalize_audio(data)

        Zxx = perform_stft(data, sample_rate, config.WINDOW_SIZE, config.HOP_LENGTH)

        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=config.WINDOW_SIZE)
        equations = []
        for i in range(Zxx.shape[1]):
            equation_components = create_equation(frequencies, Zxx[:, i])
            equations.append(equation_components)

        print(f"Created {len(equations)} equations to represent the song.")

        output_dir = getattr(config, 'OUTPUT_DIR', os.path.dirname(config.INPUT_SONG))
        base_name = os.path.splitext(os.path.basename(config.INPUT_SONG))[0]

        equations_path = os.path.join(output_dir, f"{base_name}_equations.txt")
        save_equations_to_text(equations, equations_path)

        config.INPUT_FORMULA = equations_path

        print(f"Song converted to formulas. Use formula_to_song.py to reconstruct the audio.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
