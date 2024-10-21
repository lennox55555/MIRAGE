# Music-Mask-Adversarial-Attack

Resources:

Main Folder: 

https://drive.google.com/drive/folders/1HCV-JvT3qGOSIKYPMTZ3BZDMC3YK1YfV?usp=drive_link

https://docs.google.com/document/d/13r4rJeDVK2oXUU2blaCPFalAY5RN2bQUMBlERZ3LDJ0/edit?pli=1

XAI Project Thought & Task List:
link to iphone notes app

# To Download and Run the RVC Model Locally for Yourself:

1. Download the Pretrained RVC WebUI file [here](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z)

2. If Running on Mac Silicon, MPS currently does not work with RVC. However they have not updated the package to use CPU. To do this manually open config.py located in the root directory and update the function has_mps() to return False

3. Open a terminal and create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

4. Install the required dependencies
```
pip install -r requirements.txt
```

5. Run the WebUI
```
sh ./run.sh      
```

[tutorial on how to install and use RVC](https://www.youtube.com/watch?v=qZ12-Vm2ryc&ab_channel=p3tro)