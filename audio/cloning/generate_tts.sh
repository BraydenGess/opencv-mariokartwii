#!/bin/bash

# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "audio/cloning/audio_GusJohnson01-01.wav" \
--ref_text "That place is loud and it's so aesthetically beautiful. Your right there on
the water, and washington has those cool uniforms with orange I mean uh a purple,
black and yellow" \
--gen_text "Thanks Jenny! Snow is always fun for the fans but how will it impact today's race?"