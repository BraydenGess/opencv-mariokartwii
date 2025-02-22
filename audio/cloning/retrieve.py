import simpleaudio as sa

wave_obj = sa.WaveObject.from_wave_file("audio/cloning/speech.wav")
#wave_obj = sa.WaveObject.from_wave_file("audio/cloning/GusJohnson/wavs/GusJohnson005.wav")


play_obj = wave_obj.play()

play_obj.wait_done()


