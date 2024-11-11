from rvc_python.infer import RVCInference
import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class MIRAGE_RVC:
    def __init__(self, models_dir="./voice_models", device="cpu:0", model_versions={}):
        self.rvc = RVCInference(device=device)
        self.model_versions=model_versions

        self.rvc.set_models_dir(models_dir)
    
    def run_rvc(self, model_name, input_file, output_file, **kwargs):
        '''
        run_rvc: converts an input audio file into a voice that sounds like the one specified in model_name

        Inputs:
            - model_name (str): the name of the voice to be changed into
            - input_file (str): the path to the input file
            - output_file (str): the path to the output file (must be a .wav)
            - **kwargs (dict): the parameters to tune the model conversion. Parameter Options:
                    f0method: the Pitch Extraction Method (defaults to rmvpe)
                    f0up_key: the value to change the pitch by (+12 = up the pitch one octave)
                    index_rate: Search feature ratio (defaults to 0.6)
                    filter_radius: Median filtering to the pitch results (defaults to 3)
                    resample_sr: Resample rate for the output audio (defaults to 0 = doesn't apply)
                    rms_mix_rate: Volume envelope mix rate (defaults to 0.25)
                    protect: Protect voiceless consonants and breath sounds (defaults to 0.5 aka inactive)
                *Note: RVC WebUI has explanations for what these parameters do
        '''
        self.rvc.set_params(**kwargs)
        self.rvc.load_model(model_name, version=self.model_versions[model_name])
        self.rvc.infer_file(input_file, output_file)

def main():
    model_versions = { 
        "21savage": "v1", 
        "IceSpice": "v1", 
        "2pac": "v1", 
        "jcole_100Epochs": "v2"
    }

    rvc_model = MIRAGE_RVC(model_versions=model_versions)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_input = os.path.join(script_dir, 'input_audios')
    default_output = os.path.join(script_dir, 'output_audios')
    default_models = os.path.join(script_dir, 'voice_models')

    parser = argparse.ArgumentParser(description="Split audio files into segments")
    parser.add_argument("--input", help="Path to the folder with input audio files", default=default_input)
    parser.add_argument("--output", help="Path to the folder for output audio files", default=default_output)
    parser.add_argument("--models", help="Path to the folder for output audio files", default=default_models)
    parser.add_argument("--model", help="The name of the model you want to use", default="")
    parser.add_argument("--pitch", type=int, help="value to change the of the pitch scaling", default=0)
    parser.add_argument("--protect", type=int, help="Minimum duration in seconds", default=0.33)
    parser.add_argument("--rms_rate", type=int, help="Volume envelope mix rate ", default=0.75)

    args = parser.parse_args()
    model_params = {
        'f0up_key': args.pitch,
        'protect': args.protect,
        'rms_mix_rate': args.rms_rate,
    }

    input_files = args.input
    files = [f for f in os.listdir(input_files) if os.path.isfile(os.path.join(input_files, f)) and not f.startswith('.DS_Store')]
    for audio in files:
        rvc_model.run_rvc(args.model, os.path.join(input_files, audio), os.path.join(args.output, audio), **model_params)

if __name__ == '__main__':
    main()