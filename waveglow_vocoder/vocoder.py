import json
import os

import numpy as np
import torch

import waveglow_vocoder.glow
from waveglow_vocoder.audio_layers import STFT, Denoiser, TacotronSTFT


class WaveGlowVocoder(object):
    def __init__(self, config_path="config.json", waveglow_path="waveglow_256channels_universal_v5.pt"):
        with open(config_path) as f:
            data = f.read()
        config = json.loads(data)
        if os.path.exists(waveglow_path):
            waveglow = torch.load(waveglow_path)['model']
        else:
            print(f"WaveGlow pretrained model path={waveglow_path} not exists, downloading from torchhub.")
            waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')

        # Prepare the waveglow model for inference
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to('cuda')
        self.denoiser = Denoiser(waveglow, mode='zeros').cuda()
        self.waveglow = waveglow.eval()

        # Doing Mel with TacotronSTFT as Encoder, parameter are from config.json of WaveGlow training
        data_config = config["data_config"]
        data_config.pop('training_files', None)
        data_config.pop('segment_length', None)
        self.stft = TacotronSTFT(**data_config).cuda()

    def wav2mel(self, audio):
        # Input shape: (batch_size, wav_sample_len)
        # Output shape: (batch_size, num_mel, num_window)
        audio_norm = audio.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        return melspec

    def mel2wav(self, mel, denoise=0.001):
        # Input shape: (batch_size, num_mel, num_window)
        # Output shape: (batch_size, wav_sample_len)
        with torch.no_grad():
            audio = self.waveglow.infer(mel)
            audio = self.denoiser(audio, denoise)
        return audio.squeeze(1)

if __name__ == "__main__":
    import librosa
    # Load example wav file
    y,sr = librosa.load(librosa.util.example_audio_file(), sr=22050, mono=True, duration=10, offset=30)
    y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)

    # You can apply mel transform and decoder it with Waveglow
    WV = WaveGlowVocoder()
    mel = WV.wav2mel(y_tensor)
    wav = WV.mel2wav(mel)
    # get shape: (batch_size, wav_sample_len)
    wav = wav[0].cpu().numpy()

    # Save waveform
    librosa.output.write_wav("waveglow.wav", wav, sr)
    librosa.output.write_wav("original.wav", y, sr)
