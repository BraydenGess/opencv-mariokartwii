#!/bin/bash

# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "audio/cloning/audio_GusJohnson01.wav" \
--ref_text "That place is loud and it's so aesthetically beautiful. Your right there on
the water, and washington has those cool uniforms with orange I mean uh a purple,
black and yellow" \
--gen_text "Moonview Highway is the real X factor. Do you trust the liver of the young guys
or the experience of the veterans?"