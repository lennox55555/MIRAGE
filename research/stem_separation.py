import numpy as np
import librosa
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def load_audio(file_path, sr=None):
    """Load an audio file in stereo."""
    try:
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=False)
        if audio.ndim == 1:
            audio = np.repeat(audio[np.newaxis, :], 2, axis=0)
        return audio, orig_sr
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {str(e)}")

def separate_audio(model, audio):
    """Separate the audio using the Demucs model."""
    try:
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu')
        return sources.cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"Error separating audio: {str(e)}")

def save_audio(audio, sr, file_path):
    """Save audio to a file."""
    try:
        sf.write(file_path, audio.T, sr)
    except Exception as e:
        raise RuntimeError(f"Error saving audio file: {str(e)}")

def process_song(input_file, output_dir=None, target_sr=44100):
    try:
        print(f"Loading audio file: {input_file}")
        audio, sr = load_audio(input_file, sr=target_sr)

        model = get_model('htdemucs')
        model.cpu()
        model.eval()

        print("Separating audio...")
        sources = separate_audio(model, audio)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        if output_dir is None:
            output_dir = os.path.dirname(input_file)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print("Saving separated audio files...")
        source_names = ['drums', 'bass', 'other', 'vocals']
        for i, source in enumerate(source_names):
            output_path = os.path.join(output_dir, f"{base_name}_{source}.wav")
            save_audio(sources[0, i], sr, output_path)
            print(f"{source.capitalize()} track saved to: {output_path}")

    except Exception as e:
        print(f"Error processing song: {str(e)}")

def main():
    try:
        input_file = config.INPUT_SONG
        output_dir = getattr(config, 'OUTPUT_DIR', None)
        target_sr = config.TARGET_SR

        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Target sample rate: {target_sr}")

        process_song(input_file, output_dir, target_sr)
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()