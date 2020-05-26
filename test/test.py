import librosa
import torch
from waveglow_vocoder import WaveGlowVocoder

y,sr = librosa.load(librosa.util.example_audio_file(), sr=22050, mono=True, duration=10, offset=30)
y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)

WV = WaveGlowVocoder()
mel = WV.wav2mel(y_tensor)
wav = WV.mel2wav(mel)
wav = wav[0].cpu().numpy()

WV = WaveGlowVocoder(config_path="./load_test/config.json")
mel = WV.wav2mel(y_tensor)
wav = WV.mel2wav(mel)
wav = wav[0].cpu().numpy()

WV = WaveGlowVocoder(config_path="./load_test/config.json", waveglow_path="./load_test/waveglow_256channels_universal_v5.pt")
mel = WV.wav2mel(y_tensor)
wav = WV.mel2wav(mel)
wav = wav[0].cpu().numpy()

WV = WaveGlowVocoder(waveglow_path="./load_test/waveglow_256channels_universal_v5.pt")
mel = WV.wav2mel(y_tensor)
wav = WV.mel2wav(mel)
wav = wav[0].cpu().numpy()

WV = WaveGlowVocoder(config_path="./load_test/config.json", waveglow_path="./load_test/waveglow_256channels_universal_v5.pt", device="cpu")
mel = WV.wav2mel(y_tensor.cpu())
wav = WV.mel2wav(mel)
wav = wav[0].cpu().numpy()

WV = WaveGlowVocoder(device="cpu")
mel = WV.wav2mel(y_tensor.cpu())
wav = WV.mel2wav(mel)
wav = wav[0].cpu().numpy()

librosa.output.write_wav("music_waveglow.wav", wav, sr)
librosa.output.write_wav("music_original.wav", y, sr)
