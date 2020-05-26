import json
import os
import warnings
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import waveglow_vocoder.glow as glow
from waveglow_vocoder.audio_layers import STFT, Denoiser, TacotronSTFT

class WaveGlowVocoder(object):
    def __init__(self, config_path=None, waveglow_path="waveglow_256channels_universal_v5.pt", device=None):
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if config_path == None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
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
        # Output shape: (batch_size, num_mel, num_window)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) >= 2:
            assert("Input shape: (wav_sample_len,) or (batch_size, wav_sample_len)")
        
        if torch.max(torch.abs(audio)) > 1.0:
            warnings.warn('Maximum amplitude of input waveform over 1.0, clipping.')
            audio[audio<-1.0] = -1.0
            audio[audio>1.0] = 1.0

        melspec = self.stft.mel_spectrogram(audio)
        return melspec

    def mel2wav(self, mel, denoise=0.001):
        # Input shape: (batch_size, num_mel, num_window)
        # Output shape: (batch_size, wav_sample_len)
        with torch.no_grad():
            audio = self.waveglow.infer(mel)
            audio = self.denoiser(audio, denoise)
        return audio.squeeze(1)

