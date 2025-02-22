import os
from pydub import AudioSegment

# Path to the GusJohnson directory
input_directory = "audio/cloning/GusJohnson/wavs/"
output_directory = "audio/cloning/GusJohnson/wavs_mono/"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all WAV files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Load the stereo WAV file
        stereo_sound = AudioSegment.from_wav(input_path)

        # Convert to mono
        mono_sound = stereo_sound.set_channels(1)

        # Export the mono WAV file
        mono_sound.export(output_path, format="wav")

        print(f"Converted {filename} to mono.")

