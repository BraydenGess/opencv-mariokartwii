import os

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
# we use the same path as this script as our training folder.
    output_path = '/Users/bradygess/PycharmProjects/opencv-mariokartwii/audio/cloning'

    # DEFINE DATASET CONFIG
    # Set LJSpeech as our target dataset and define its path.
    # You can also use a simple Dict to define the dataset and pass it to your custom formatter.
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "GusJohnson/")
    )

    # INITIALIZE THE TRAINING CONFIGURATION
    # Configure the model. Every config class inherits the BaseTTSConfig.
    config = GlowTTSConfig(
        batch_size=2,
        eval_batch_size=2,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=50,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )

    ap = AudioProcessor.init_from_config(config)


    tokenizer, config = TTSTokenizer.init_from_config(config)


    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.4,
    )

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)


    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # AND... 3,2,1... 🚀
    trainer.fit()

### command line
## tts --text "Sample text" --model_path path/to/best_model.pth
# --config_path path/to/config.json --out_path path/to/speech.wav

if __name__ == "__main__":
    main()