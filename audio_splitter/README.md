### To Run Audio Splitter

### Required Dependencies
pydub==0.25.1

### To Run
1. Create two folders inside the audio_splitter folder 
(assuming you are running a terminal instance from the Music-Mask-Adversarial-Attack)
```
mkdir audio_splitter/input_audios audio_splitter/output_audios
```

2. Run Script:
```
python audio_splitter/audio_splitter.py
```

### Notes
- --input = can specify input folder
- --output = can specify output folder
- --split = can specify length of segments