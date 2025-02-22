Create virtual environment  
python3 -m venv tts_env  
source tts_env/bin/activate  
pip install TTS  

Dataset is at metadata.csv  
Should be in format filename|text|transcription  
Formatting:  
Audio Format: WAV (PCM)  
Sample Rate: 22050  
Channels: Mono  
Resolution: 16-bit PCM  
Dithering: None (preferred) or Triangular  

python train_clone.py (train the model)  

$ tts --text "Sample Text" --model_path path/to/best_model.pth   
--config_path path/to/config.json --out_path path/to/output.wav (load best model)  

retrieve.py (run the model)  

Data collection:  
Download youtube videos with wav converter  
ocenaudio for easy clipping of the wav file to bits  