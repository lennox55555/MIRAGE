# MIRAGE

**A Comprehensive Audio Analysis Pipeline**

---

## Overview

MIRAGE is a Python-based pipeline designed for researchers to analyze, modify, and compare audio files, with a particular focus on music. This pipeline uses Demucs for stem separation, Librosa for feature extraction, and SQL-based data storage to enable high-level data organization and retrieval. MIRAGE is versatile and equipped for tasks ranging from creating formulas to reconstructing audio files from formula-based transformations.

## Database Structure

MIRAGE stores song information and extracted features in a SQLite database for easy access and management.

### Songs Table:

| id | artist_name | song_name   |
|----|-------------|-------------|
| 1  | Artist Name | Song Title  |

Each song entry is unique by artist name and song title. MIRAGE checks for duplicate entries before adding a new song to the database.

### Features Table:

| song_id | stem   | feature_name | feature_values          |
|---------|--------|--------------|-------------------------|
| 1       | vocals | mfccs        | -123.45, -120.67, ...   |
| 1       | vocals | chroma       | 0.12, 0.34, ...         |
| 1       | drums  | mfccs        | -130.56, -125.78, ...   |
| ...     | ...    | ...          | ...                     |

Each row corresponds to a specific feature (such as `mfccs` or `mel_spectrogram`) of a particular stem (e.g., vocals, bass, or drums). Feature values are stored as serialized arrays to enable efficient data access and comparison.

## Feature Dimensionality

For each stem, MIRAGE extracts the following features:

- **MFCCs (Mel-Frequency Cepstral Coefficients)**: 20 coefficients to capture spectral characteristics.
- **Chroma**: 12 chroma bins representing the 12 semitones in an octave.
- **Spectral Contrast**: 7 coefficients capturing contrast between peaks and valleys in each sub-band.
- **Tonnetz**: A 6-dimensional representation of tonal centroid features.

Additional features such as mel spectrogram are available and stored as needed. The dimensionality of each feature type is determined by standard parameters used in `librosa`.

## Pipeline Options and Functionalities

Upon running the pipeline, the user is prompted to select one of the following options:

### Option 1: Add to Database

1. **User Inputs**: Artist name, song title, and file path.
2. **Process**:
   - The audio file is loaded, normalized, and saved in stereo format.
   - MIRAGE extracts selected features (MFCCs, chroma, mel spectrogram) for both the full song and individual stems (vocals, drums, bass, other).
   - Checks for duplicate songs based on artist name, song title, and feature similarity. If a song is nearly identical to an existing song, it only performs stem separation without adding a duplicate entry.
3. **Output**:
   - Original audio and stems saved in organized directories.
   - OSCR metric (original and reconstructed song similarity) calculated and displayed.
   - Song data, stem features, and computed metrics saved to the database.

### Option 2: Read from Database

Displays all songs and features currently stored in the database. This option helps researchers quickly access the stored audio files and their associated features.

### Option 3: Get Cosine Similarity

1. **User Prompts**:
   - User selects two songs from the database.
2. **Process**:
   - For each selected feature (mel spectrogram, MFCCs, chroma, spectral contrast, and tonnetz), MIRAGE calculates the cosine similarity between the corresponding stems.
3. **Output**:
   - Cosine similarity scores for each feature are displayed, allowing users to quantify similarities between two songs at various levels of the audio structure.

### Option 4: Combine Stems

1. **User Prompts**:
   - Select a song to reconstruct.
   - If the stems are unavailable, the user is prompted to split them.
   - User selects which stems (e.g., vocals, drums, bass) to combine into a new audio file.
2. **Process**:
   - MIRAGE combines the selected stems and saves the reconstructed audio in a custom directory.
3. **Output**:
   - A combined audio file is saved, allowing researchers to analyze specific portions of the song or create new audio samples.

### Option 5: Convert Song to Formula

1. **User Prompts**:
   - Select a song to convert.
2. **Process**:
   - MIRAGE performs a Short-Time Fourier Transform (STFT) on the audio file to represent each time slice as a formula based on frequency, magnitude, and phase.
   - The resulting equations are stored in a text file.
3. **Output**:
   - A directory containing the song's formula text file. This file serves as an analytical representation of the song's structure and allows researchers to synthesize the song later.

### Option 6: Convert Formula to Synthesized Song

1. **User Prompts**:
   - Select a formula text file to synthesize.
2. **Process**:
   - MIRAGE uses the saved formula to recreate the song by reconstructing the original waveform.
3. **Output**:
   - A synthesized version of the song is saved in the designated directory.

## Directory Structure

- **output/original**: Contains original versions of all added songs.
- **output/stems**: Stores the separated stems (vocals, drums, bass, other) for each song.
- **output/reconstructed**: Contains audio reconstructed from the saved stems.
- **output/formulas**: Stores formula-based representations of songs as text files.
- **output/synthesized**: Contains synthesized versions of songs recreated from formulas.
- **output/custom merged stems**: Stores custom reconstructions where the user has combined specific stems.

## Resources and External Links

- **Main Folder**: [Google Drive Link](https://drive.google.com/drive/folders/1HCV-JvT3qGOSIKYPMTZ3BZDMC3YK1YfV?usp=drive_link)
- **Documentation**: [MIRAGE Documentation](https://docs.google.com/document/d/13r4rJeDVK2oXUU2blaCPFalAY5RN2bQUMBlERZ3LDJ0/edit?pli=1)
- **XAI Project Thought & Task List**: Access this on the iPhone Notes app (if shared).

