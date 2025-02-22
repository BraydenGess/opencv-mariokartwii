import simpleaudio as sa

def play_audio():
    #wave_obj = sa.WaveObject.from_wave_file("audio/cloning/speech.wav")
    wave_obj = sa.WaveObject.from_wave_file("tests/infer_cli_basic.wav")
    #wave_obj = sa.WaveObject.from_wave_file("audio/cloning/GusJohnson/wavs/GusJohnson01-13.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

def generate_audio():
    from TTS.api import TTS
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2',gpu=False)

    tts.tts_to_file(text='Welcome back to Luigi Circuit. It is a great day for racing.',
                file_path = 'audio/cloning/speech.wav',
                speaker_wav = ['audio/cloning/GusJohnson009'],
                language = 'en',
                split_sentences=True,
                weights_only = False)

play_audio()

