import librosa
import soundfile as sf
import torch

from waveglow_vocoder import WaveGlowVocoder

# Load example wav file
y,sr = librosa.load(librosa.util.example(key='brahms'), sr=22050, mono=True, duration=10, offset=30)
y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)

# You can apply mel transform and decoder it with Waveglow
WV = WaveGlowVocoder()
mel = WV.wav2mel(y_tensor)
wav = WV.mel2wav(mel)
# get shape: (batch_size, wav_sample_len)
wav = wav[0].cpu().numpy()

# Save waveform
sf.write("music_waveglow.wav", wav, sr)
sf.write("music_original.wav", y, sr)