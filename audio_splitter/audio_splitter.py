import os
import argparse
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def split_audio(input_wav, file_name, output_dir, silence_thresh=-40.0, min_duration=5000, min_silence_duration=500, max_segment_length=10000):
    """
    Removes silence and splits audio into chunks, saving each chunk as a separate file.

    Parameters:
        input_wav (str): Path to the input WAV file.
        output_dir (str): Directory where the output chunks will be saved.
        silence_thresh (float): Silence threshold in dBFS. Default is -40.0 dBFS.
        min_silence_duration (int): Minimum duration of silence in milliseconds to consider as silence. Default is 500 ms.
        max_segment_length (int): Maximum length of each segment in milliseconds. Default is 10 seconds (10000 ms).
    
    Returns:
        None: The function saves each chunk as a separate WAV file in the output directory.
    """
    # Load the audio using pydub
    audio = AudioSegment.from_wav(input_wav)

    # Detect non-silent chunks (start, end in milliseconds)
    nonsilent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_duration, silence_thresh=silence_thresh)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize list to hold audio segments
    audio_segments = []

    # Process each non-silent chunk
    for start, end in nonsilent_chunks:
        chunk_length = end - start
        
        # Split chunks longer than max_segment_length into smaller segments
        if chunk_length > max_segment_length:
            num_full_segments = chunk_length // max_segment_length
            for i in range(num_full_segments):
                segment_start = start + i * max_segment_length
                segment_end = segment_start + max_segment_length
                
                if segment_end - segment_start >= min_duration:
                    audio_segments.append(audio[segment_start:segment_end])
            
            # Add the remaining part if it's more than 0 ms
            remaining_start = start + num_full_segments * max_segment_length
            if remaining_start + min_duration <= end:
                audio_segments.append(audio[remaining_start:end])
        elif chunk_length >= min_duration:
            # Add the chunk directly if it is 10 seconds or less
            audio_segments.append(audio[start:end])

    # Export each segment as a separate file
    for i, segment in enumerate(audio_segments):
        segment_filename = os.path.join(output_dir, f"{file_name}_{i+1}.wav")
        segment.export(segment_filename, format="wav")

def split_audios(input_folder, output_folder, min_duration, split):
    '''
    Splits an audio file into equal segments based on a time

    Inputs:
        - input_folder (str): the path to the input audio files
        - output_folder (str): the path to the output audio files
        - split (int): how long each split should be (in seconds)
    '''
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and not f.startswith('.DS_Store')]

    for file in files:
        split_audio(os.path.join(input_folder, file), os.path.splitext(file)[0], output_folder, min_duration=min_duration*1000, max_segment_length=split*1000)

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_input = os.path.join(script_dir, 'input_audios')
    default_output = os.path.join(script_dir, 'output_audios')

    parser = argparse.ArgumentParser(description="Split audio files into segments")
    parser.add_argument("--input", help="Path to the folder with input audio files", default=default_input)
    parser.add_argument("--output", help="Path to the folder for output audio files", default=default_output)
    parser.add_argument("--split", type=int, help="Segment duration in seconds", default=10)
    parser.add_argument("--duration", type=int, help="Minimum duration in seconds", default=5)

    args = parser.parse_args()
    split_audios(args.input, args.output, args.duration, args.split)

if __name__ == '__main__':
    main()