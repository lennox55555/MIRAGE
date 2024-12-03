### To Run RVC Programmatic

### Required Dependencies
rvc-python==0.1.4
tensorboardX==2.6.2.2

### To Run
- Create three folders inside the rvc_programmatic folder 
    1. input_audios - the audio files you want as input
    2. output_audios - where the output audio files will be
    3. voice_models - the folder that will contain all the pretrained models

- Add the name of the folder for the voice model you want to use to the model_versions dictionary in run_rvc.py It must be of the form {"model_name": "version" } where version is either "v1" or "v2" depending on how the model was trained. This will lead to an error if the version is wrong.

### If Running on Mac
- Go to ./Music-Mask-Adversarial-Attack/venv/lib/python3.9/site-packages/rvc_python/configs/config.py and change the function has_mps() to return False

- After installing the required dependencies cd into the rvc_programmatic folder. Now you can run:
```
python run_rvc.py --model {model name from dictionary} 
```
Example:
```
python run_rvc.py --model  
```

### Structure of the voice_models folder:
voice_models/
├── {name_of_artist}/
│   ├── model.pth file
│   └── model.index file


### Notes
- First run will take a while because it needs to download the required feature extraction method .pt files