import numpy as np
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sqlite3
import os
from scipy.spatial.distance import cosine
import json
import soundfile as sf

def load_audio(file_path, sr=44100, mono=False):
    """Load an audio file and ensure it's in the correct format."""
    try:
        data, sample_rate = sf.read(file_path)
        if isinstance(data, (int, float)):
            raise ValueError("Unexpected audio data type. Ensure the file is a valid audio file.")
        if data.ndim == 1 and not mono:
            # Duplicate the mono channel to create a stereo signal
            data = np.stack((data, data), axis=-1)
        elif data.ndim > 1 and mono:
            # Convert stereo to mono
            data = np.mean(data, axis=1)
        if sample_rate != sr:
            if data.ndim == 1:
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
            else:
                data = librosa.resample(data.T, orig_sr=sample_rate, target_sr=sr).T
            sample_rate = sr
        return sample_rate, data
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {str(e)}")

def separate_audio(audio, model):
    """Separate the audio into stems using the Demucs model."""
    try:
        # Ensure audio is in shape (n_samples, n_channels)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        # Add batch dimension: (1, n_channels, n_samples)
        audio_tensor = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(model, audio_tensor)
        return sources.cpu().numpy()  # Shape: (1, n_sources, n_channels, n_samples)
    except Exception as e:
        raise RuntimeError(f"Error separating audio: {str(e)}")

def extract_features(audio, sr):
    """Extract various audio features from the audio signal."""
    features = {}
    # If audio is stereo, convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Assuming audio shape is (n_samples, n_channels)
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    else:
        print("Warning: Audio signal is zero everywhere.")
    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spectrogram'] = mel_spec_db  # Retain full mel spectrogram
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features['mfccs'] = mfccs  # Retain full MFCCs
    # Compute Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma'] = chroma  # Retain full chroma features
    # Compute Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['contrast'] = contrast  # Retain full spectral contrast
    # Compute Tonnetz
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    features['tonnetz'] = tonnetz  # Retain full tonnetz features
    return features

def process_stems(sources, sr):
    """Process each stem to extract features."""
    stem_features = {}
    source_names = ['drums', 'bass', 'other', 'vocals']
    for i, source in enumerate(source_names):
        print(f"Extracting features from {source}...")
        audio = sources[0, i].T  # Shape: (n_samples, n_channels)
        features = extract_features(audio, sr)
        stem_features[source] = features
    return stem_features

def save_stems(sources, sr, song_name, artist_name):
    """Save the separated stems to the output directory."""
    source_names = ['drums', 'bass', 'other', 'vocals']
    for i, source in enumerate(source_names):
        stem_audio = sources[0, i].T  # Shape: (n_samples, n_channels)
        # Modify filename to prevent collisions
        output_filename = f"{song_name}_{artist_name}_STEM_{source}.wav"
        output_path = os.path.join(STEMS_DIR, output_filename)
        sf.write(output_path, stem_audio, sr)
        print(f"Saved {source} stem to {output_path}")

def reconstruct_audio_from_stems(sources):
    """Reconstruct the audio by summing the stems."""
    # Sum the stems over axis=1 (over sources)
    reconstructed_audio = np.sum(sources[0], axis=0).T  # Shape: (n_samples, n_channels)
    return reconstructed_audio

def create_database(db_path='songs.db'):
    """Create a SQLite database to store songs and features."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artist_name TEXT,
            song_name TEXT,
            file_path TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            song_id INTEGER,
            stem TEXT,
            feature_name TEXT,
            feature_values TEXT,
            FOREIGN KEY(song_id) REFERENCES songs(id)
        )
    ''')
    conn.commit()
    return conn

def save_to_database(conn, artist_name, song_name, input_file, stem_features, original_features):
    """Save song information and features to the database."""
    cursor = conn.cursor()
    # Convert artist_name and song_name to lowercase
    artist_name = artist_name.lower()
    song_name = song_name.lower()
    # Check for duplicates
    cursor.execute('''
        SELECT id FROM songs WHERE artist_name = ? AND song_name = ?
    ''', (artist_name, song_name))
    result = cursor.fetchone()
    if result:
        print("This song already exists in the database. Skipping insertion.")
        return
    # Insert song
    cursor.execute('''
        INSERT INTO songs (artist_name, song_name, file_path) VALUES (?, ?, ?)
    ''', (artist_name, song_name, input_file))
    song_id = cursor.lastrowid
    # Insert features for stems
    for stem, features in stem_features.items():
        for feature_name, feature_values in features.items():
            # Serialize the feature array
            feature_values_serialized = json.dumps(feature_values.tolist())
            cursor.execute('''
                INSERT INTO features (song_id, stem, feature_name, feature_values)
                VALUES (?, ?, ?, ?)
            ''', (song_id, stem, feature_name, feature_values_serialized))
    # Insert features for original audio
    for feature_name, feature_values in original_features.items():
        # Serialize the feature array
        feature_values_serialized = json.dumps(feature_values.tolist())
        cursor.execute('''
            INSERT INTO features (song_id, stem, feature_name, feature_values)
            VALUES (?, ?, ?, ?)
        ''', (song_id, 'mix', feature_name, feature_values_serialized))
    conn.commit()

def read_database(conn):
    """Read and display the contents of the database."""
    cursor = conn.cursor()
    # Read songs table
    print("\nSongs Table:")
    cursor.execute("SELECT * FROM songs")
    songs = cursor.fetchall()
    for song in songs:
        song_id, artist_name, song_name, _ = song
        print(f"ID: {song_id}, Artist: {artist_name}, Song: {song_name}")
    # Read features table
    print("\nFeatures Table:")
    cursor.execute("SELECT * FROM features")
    features = cursor.fetchall()
    for feature in features:
        song_id, stem, feature_name, _ = feature
        print(f"Song ID: {song_id}, Stem: {stem}, Feature: {feature_name}, Values: [Serialized data]")
    print("\n")

def list_songs(conn):
    """List all songs in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, artist_name, song_name FROM songs")
    songs = cursor.fetchall()
    for song in songs:
        song_id, artist_name, song_name = song
        print(f"ID: {song_id}, Artist: {artist_name}, Song: {song_name}")
    return songs

def get_features(song_id, stem, feature_name, conn):
    """Retrieve features for a specific stem and feature name of a song from the database."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT feature_values FROM features
        WHERE song_id = ? AND feature_name = ? AND stem = ?
    ''', (song_id, feature_name, stem))
    result = cursor.fetchone()
    if not result:
        print(f"No features found for song ID {song_id}, stem '{stem}', and feature '{feature_name}'")
        return None
    feature_values_serialized = result[0]
    feature_values = np.array(json.loads(feature_values_serialized))
    return feature_values

def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    # Flatten the arrays
    a_flat = a.flatten()
    b_flat = b.flatten()
    # Ensure both arrays are the same length
    min_length = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_length]
    b_flat = b_flat[:min_length]
    # Compute cosine similarity
    cos_sim = 1 - cosine(a_flat, b_flat)
    return cos_sim

def calculate_difference_percentage(original, perturbed):
    """Compute percentage difference between original and perturbed signals."""
    # Ensure both signals are of the same length
    min_length = min(len(original), len(perturbed))
    original = original[:min_length]
    perturbed = perturbed[:min_length]
    # Calculate the absolute difference
    diff = np.abs(original - perturbed)
    # Calculate the percentage difference based on amplitude
    percentage_diff = (np.sum(diff) / np.sum(np.abs(original))) * 100
    return percentage_diff

def stems_exist(song_name, artist_name):
    """Check if stems for a song exist in the file system."""
    source_names = ['drums', 'bass', 'other', 'vocals']
    for source in source_names:
        stem_filename = f"{song_name}_{artist_name}_STEM_{source}.wav"
        stem_path = os.path.join(STEMS_DIR, stem_filename)
        if not os.path.isfile(stem_path):
            return False
    return True

def get_valid_song_id(prompt, conn):
    """Prompt the user for a valid song ID."""
    while True:
        song_id = input(prompt)
        if not song_id.isdigit():
            print("Invalid input. Please enter a numeric song ID.")
            continue
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM songs WHERE id = ?", (song_id,))
        song = cursor.fetchone()
        if not song:
            print(f"No song found with ID {song_id}. Please try again.")
        else:
            return song_id

def get_menu_choice():
    """Prompt the user for a valid menu choice."""
    while True:
        print("Choose an option:")
        print("1. Add to database")
        print("2. Read from database")
        print("3. Get cosine similarity")
        print("4. Combine stems")
        print("5. Convert song to formula")
        print("6. Convert formula to synthesized song")
        choice = input("Enter your choice (1-6): ")
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        else:
            print("Invalid choice. Please select a valid option (1-6).")

# Define output directories at the module level
OUTPUT_DIR = 'output'
ORIGINAL_DIR = os.path.join(OUTPUT_DIR, 'original')
STEMS_DIR = os.path.join(OUTPUT_DIR, 'stems')
RECONSTRUCTED_DIR = os.path.join(OUTPUT_DIR, 'reconstructed')
FORMULAS_DIR = os.path.join(OUTPUT_DIR, 'formulas')
SYNTHESIZED_DIR = os.path.join(OUTPUT_DIR, 'synthesized')

# Constants for STFT
WINDOW_SIZE = 1024
HOP_LENGTH = 512
TARGET_SR = 44100  # Target sample rate

def normalize_audio(data):
    """Normalize audio data to range [-1, 1]."""
    if np.max(np.abs(data)) > 0:
        return data / np.max(np.abs(data))
    else:
        return data

def perform_stft(data, sample_rate, window_size, hop_length):
    """Perform Short-Time Fourier Transform (STFT)."""
    return librosa.stft(data, n_fft=window_size, hop_length=hop_length, window='hann')

def create_equation(frequencies, Zxx_slice):
    """Create equation components from STFT slice."""
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
    """Save equations to a text file."""
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

def load_equations(file_path):
    """Load equations from a text file."""
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
    """Synthesize audio from equations."""
    n_fft = WINDOW_SIZE
    n_frames = len(equations)
    Zxx = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)

    for i, eq_components in enumerate(equations):
        for component in eq_components:
            freq_index = int(component["frequency"] * n_fft / sample_rate)
            if freq_index < Zxx.shape[0]:
                Zxx[freq_index, i] = component["magnitude"] * np.exp(1j * component["phase"])

    return librosa.istft(Zxx, hop_length=hop_length, window='hann')

def main():
    # Create output directories if they don't exist
    output_dirs = [OUTPUT_DIR, ORIGINAL_DIR, STEMS_DIR, RECONSTRUCTED_DIR, FORMULAS_DIR, SYNTHESIZED_DIR]
    for directory in output_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    conn = create_database()

    # Present options to the user
    choice = get_menu_choice()

    if choice == '1':
        # Add to database
        try:
            # Prompt user for inputs
            artist_name = input("Enter the artist's name: ")
            song_name = input("Enter the song's name: ")
            input_file = input("Enter the path to the song file: ")

            # Load audio without converting to mono
            print(f"Loading audio file: {input_file}")
            sample_rate, audio = load_audio(input_file, sr=TARGET_SR, mono=False)
            print(f"Audio shape before saving original: {audio.shape}")

            # Save original audio with unique filename
            original_output_path = os.path.join(ORIGINAL_DIR, f"{song_name}_{artist_name}_ORIGINAL.wav")
            sf.write(original_output_path, audio, sample_rate)
            print(f"Saved original audio to {original_output_path}")

            # Extract features from original audio
            print("Extracting features from original audio...")
            # For feature extraction, convert to mono
            mono_audio = np.mean(audio, axis=1)
            original_features = extract_features(mono_audio, sample_rate)

            # Check for similar songs
            cursor = conn.cursor()
            cursor.execute("SELECT id, artist_name, song_name FROM songs")
            existing_songs = cursor.fetchall()
            similar_song_found = False
            for song in existing_songs:
                existing_song_id, existing_artist_name, existing_song_name = song
                # Skip if same artist and song name
                if artist_name.lower() == existing_artist_name.lower() and song_name.lower() == existing_song_name.lower():
                    print("This song already exists in the database. Skipping insertion.")
                    similar_song_found = True
                    break
                # Retrieve mix features of existing song
                existing_features = get_features(existing_song_id, 'mix', 'mel_spectrogram', conn)
                if existing_features is None:
                    continue
                # Compute cosine similarity
                cos_sim = compute_cosine_similarity(original_features['mel_spectrogram'], existing_features)
                if cos_sim >= 0.99:
                    print(f"An extremely similar song is already in the database (Song ID: {existing_song_id}). Skipping insertion.")
                    similar_song_found = True
                    break
            if similar_song_found:
                # Ask if user wants to split stems
                split_choice = input("Do you want to split the stems now? (yes/no): ").strip().lower()
                if split_choice in ['yes', 'y']:
                    # Proceed to split stems and save them
                    # Load Demucs model
                    print("Loading Demucs model...")
                    model = get_model('htdemucs')
                    model.eval()
                    # Separate audio into stems
                    print("Separating audio into stems...")
                    sources = separate_audio(audio, model)
                    # Save stems
                    save_stems(sources, sample_rate, song_name, artist_name)
                    print("Stems saved successfully.")
                else:
                    print("Operation cancelled.")
                return

            # Load Demucs model
            print("Loading Demucs model...")
            model = get_model('htdemucs')
            model.eval()

            # Separate audio into stems
            print("Separating audio into stems...")
            sources = separate_audio(audio, model)

            # Save stems
            save_stems(sources, sample_rate, song_name, artist_name)

            # Extract features from each stem
            print("Extracting features from stems...")
            stem_features = process_stems(sources, sample_rate)

            # Reconstruct audio from stems
            print("Reconstructing audio from stems...")
            reconstructed_audio = reconstruct_audio_from_stems(sources)

            # Save reconstructed audio with unique filename
            reconstructed_output_path = os.path.join(RECONSTRUCTED_DIR, f"{song_name}_{artist_name}_RECONSTRUCTED.wav")
            sf.write(reconstructed_output_path, reconstructed_audio, sample_rate)
            print(f"Saved reconstructed audio to {reconstructed_output_path}")

            # Compute OSCR metric between original and reconstructed audio
            oscr = calculate_difference_percentage(audio.flatten(), reconstructed_audio.flatten())
            print(f"OSCR metric between original and reconstructed audio: {oscr:.2f}%")

            # Save song information and features to the database
            print("Saving to database...")
            save_to_database(conn, artist_name, song_name, input_file, stem_features, original_features)
            print("Data saved to database successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

    elif choice == '2':
        # Read from database
        try:
            print("Reading from database...")
            read_database(conn)
        except Exception as e:
            print(f"An error occurred: {e}")

    elif choice == '3':
        # Get cosine similarity
        try:
            print("Available songs:")
            list_songs(conn)
            # Get song IDs from the user
            song_id1 = get_valid_song_id("Enter the ID of the first song: ", conn)
            song_id2 = get_valid_song_id("Enter the ID of the second song: ", conn)

            # Prompt the user to select the stem to compare
            valid_stems = ['vocals', 'mix', 'bass', 'drums', 'other']
            selected_stem = None
            while True:
                print("\nSelect the stem to compare:")
                for i, stem in enumerate(valid_stems, 1):
                    print(f"{i}. {stem.capitalize()}")
                stem_choice = input(f"Enter the number of your choice (1-{len(valid_stems)}): ")
                try:
                    stem_index = int(stem_choice) - 1
                    if stem_index < 0 or stem_index >= len(valid_stems):
                        raise ValueError
                    selected_stem = valid_stems[stem_index]
                    break
                except ValueError:
                    print("Invalid choice. Please select a valid stem.")

            # List of features to compare
            valid_features = ['mel_spectrogram', 'mfccs', 'chroma', 'contrast', 'tonnetz']

            # Compute and display cosine similarity for all features
            print(f"\nComputing cosine similarity for all features of the '{selected_stem}' stem:")
            for feature_name in valid_features:
                # Retrieve features from database
                features1 = get_features(song_id1, selected_stem, feature_name, conn)
                features2 = get_features(song_id2, selected_stem, feature_name, conn)
                if features1 is None or features2 is None:
                    print(f"Cannot compute cosine similarity for '{feature_name}' due to missing features.")
                    continue

                # Compute cosine similarity between feature arrays
                cos_sim_features = compute_cosine_similarity(features1, features2)
                print(f"Cosine similarity for '{feature_name}': {cos_sim_features:.4f}")
        except Exception as e:
            print(f"An error occurred: {e}")

    elif choice == '4':
        # Combine stems
        try:
            # Display available songs
            print("Available songs:")
            list_songs(conn)
            # Get song ID from the user
            song_id = get_valid_song_id("Enter the ID of the song you want to reconstruct: ", conn)
            # Retrieve song details from the database
            cursor = conn.cursor()
            cursor.execute("SELECT song_name, artist_name, file_path FROM songs WHERE id = ?", (song_id,))
            song_data = cursor.fetchone()
            if not song_data:
                print("Song not found in the database. Please add the song to the database first.")
                return
            song_name, artist_name, file_path = song_data
            # Check if stems exist
            if not stems_exist(song_name, artist_name):
                print("Stems for this song are not available.")
                # Ask user if they want to split the stems
                split_choice = input("Do you want to split the stems now? (yes/no): ").strip().lower()
                if split_choice in ['yes', 'y']:
                    # Ask for the path to the song file if not available
                    if not os.path.isfile(file_path):
                        file_path = input("Enter the path to the song file: ").strip()
                        if not os.path.isfile(file_path):
                            print("Invalid file path. Cannot proceed.")
                            return
                    # Load audio without converting to mono
                    print(f"Loading audio file: {file_path}")
                    sample_rate, audio = load_audio(file_path, sr=TARGET_SR, mono=False)
                    # Load Demucs model
                    print("Loading Demucs model...")
                    model = get_model('htdemucs')
                    model.eval()
                    # Separate audio into stems
                    print("Separating audio into stems...")
                    sources = separate_audio(audio, model)
                    # Save stems
                    save_stems(sources, sample_rate, song_name, artist_name)
                    print("Stems saved successfully.")
                else:
                    print("Cannot proceed without stems.")
                    return
            else:
                print("Stems already exist for this song.")
            # Prompt user to select stems to combine
            source_names = ['drums', 'bass', 'other', 'vocals']
            selected_stems = []
            print("\nSelect stems to combine:")
            for i, source in enumerate(source_names, 1):
                print(f"{i}. {source.capitalize()}")
            print("Enter the numbers of the stems you want to combine, separated by commas (e.g., 1,3):")
            stem_choices = input("Your choices: ").strip()
            stem_indices = stem_choices.split(',')
            try:
                for index in stem_indices:
                    idx = int(index.strip()) - 1
                    if idx < 0 or idx >= len(source_names):
                        print(f"Invalid stem selection: {index}")
                        return
                    selected_stems.append(source_names[idx])
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                return
            if not selected_stems:
                print("No stems selected. Cannot proceed.")
                return
            # Load selected stems
            stem_audios = []
            sr = None
            for stem in selected_stems:
                stem_filename = f"{song_name}_{artist_name}_STEM_{stem}.wav"
                stem_path = os.path.join(STEMS_DIR, stem_filename)
                if os.path.isfile(stem_path):
                    sample_rate, stem_audio = load_audio(stem_path, sr=TARGET_SR, mono=False)
                    stem_audios.append(stem_audio)
                else:
                    print(f"Stem file not found: {stem_path}")
                    return
            # Combine stems
            combined_audio = np.sum(stem_audios, axis=0)
            # Save combined audio
            CUSTOM_MERGED_DIR = os.path.join(OUTPUT_DIR, 'custom merged stems')
            if not os.path.exists(CUSTOM_MERGED_DIR):
                os.makedirs(CUSTOM_MERGED_DIR)
            output_filename = f"{song_name}_{artist_name}_CUSTOM_MERGED.wav"
            output_path = os.path.join(CUSTOM_MERGED_DIR, output_filename)
            sf.write(output_path, combined_audio, sample_rate)
            print(f"Combined stems saved to: {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    elif choice == '5':
        # Convert song to formula
        try:
            # List songs in the original directory
            songs = [f for f in os.listdir(ORIGINAL_DIR) if os.path.isfile(os.path.join(ORIGINAL_DIR, f))]
            if not songs:
                print("No songs found in the original directory.")
                return
            print("Available songs:")
            for i, song in enumerate(songs, 1):
                print(f"{i}. {song}")
            song_choice = input("Enter the number of the song to convert to formula: ").strip()
            try:
                song_index = int(song_choice) - 1
                if song_index < 0 or song_index >= len(songs):
                    print("Invalid selection.")
                    return
                song_filename = songs[song_index]
            except ValueError:
                print("Invalid input.")
                return
            song_path = os.path.join(ORIGINAL_DIR, song_filename)
            print(f"Selected song: {song_filename}")

            # Load audio and convert to mono for processing
            sample_rate, data = load_audio(song_path, sr=TARGET_SR, mono=True)
            data = normalize_audio(data)

            # Perform STFT
            Zxx = perform_stft(data, sample_rate, WINDOW_SIZE, HOP_LENGTH)

            # Get frequencies
            frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=WINDOW_SIZE)
            equations = []
            for i in range(Zxx.shape[1]):
                equation_components = create_equation(frequencies, Zxx[:, i])
                equations.append(equation_components)

            print(f"Created {len(equations)} equations to represent the song.")

            # Save equations to text file
            base_name = os.path.splitext(song_filename)[0]
            equations_path = os.path.join(FORMULAS_DIR, f"{base_name}_equations.txt")
            save_equations_to_text(equations, equations_path)
            print(f"Song converted to formula and saved at: {equations_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    elif choice == '6':
        # Convert formula to synthesized song
        try:
            # List formula text files in the formulas directory
            formula_files = [f for f in os.listdir(FORMULAS_DIR) if f.endswith('_equations.txt')]
            if not formula_files:
                print("No formula files found in the formulas directory.")
                return
            print("Available formula files:")
            for i, formula_file in enumerate(formula_files, 1):
                print(f"{i}. {formula_file}")
            formula_choice = input("Enter the number of the formula file to synthesize: ").strip()
            try:
                formula_index = int(formula_choice) - 1
                if formula_index < 0 or formula_index >= len(formula_files):
                    print("Invalid selection.")
                    return
                formula_filename = formula_files[formula_index]
            except ValueError:
                print("Invalid input.")
                return
            formula_path = os.path.join(FORMULAS_DIR, formula_filename)
            print(f"Selected formula file: {formula_filename}")

            # Load equations from the file
            equations = load_equations(formula_path)
            print(f"Loaded {len(equations)} equations.")

            # Synthesize audio from equations
            sample_rate = TARGET_SR
            synthesized_audio = synthesize_audio_from_equations(equations, sample_rate, HOP_LENGTH)
            synthesized_audio = normalize_audio(synthesized_audio)

            # Save synthesized audio
            base_name = os.path.splitext(formula_filename)[0]
            synth_audio_path = os.path.join(SYNTHESIZED_DIR, f"{base_name}_synthesized.wav")
            sf.write(synth_audio_path, synthesized_audio, sample_rate)
            print(f"Synthesized audio saved to: {synth_audio_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print("Invalid choice. Please select a valid option.")

    conn.close()
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
