#!/bin/bash

# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "audio/cloning/audio_JoelKlatt01-01.wav" \
--ref_text "One more player left on the offense that can go out for a passing route eligibility which is this running back
right here this running back is going to be covered in tandem by the two linebackers based on where he releases" \
--gen_text "Yes, I really like the 2 on 2 style because it adds another dimension to the race.
In order to win today both teammates are going to have to perform at a high level. One teammate is going to be worse, that's just the reality of the situation.
Today is not just about drinking and driving but also dealing with adversity and team chemistry"