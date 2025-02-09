import librosa
import soundfile as sf
import torch
import time
from waveglow_vocoder import WaveGlowVocoder

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timer
def test_default_configuration(y_tensor, sr):
    print('\nTesting WaveGlowVocoder with default configuration')
    WV = WaveGlowVocoder()
    mel = WV.wav2mel(y_tensor)
    wav = WV.mel2wav(mel)
    wav = wav[0].cpu().numpy()
    sf.write("music_waveglow_default.wav", wav, sr)

@timer
def test_with_config_file(y_tensor, sr):
    print('\nTesting WaveGlowVocoder using config file.')
    WV = WaveGlowVocoder(config_path="./load_test/config.json")
    mel = WV.wav2mel(y_tensor)
    wav = WV.mel2wav(mel)
    wav = wav[0].cpu().numpy()
    sf.write("music_waveglow_from_config.wav", wav, sr)

@timer
def test_with_waveglow_path(y_tensor, sr):
    print('\nTesting WaveGlowVocoder using waveglow_path.')
    WV = WaveGlowVocoder(waveglow_path="./load_test/waveglow_256channels_universal_v5.pt")
    mel = WV.wav2mel(y_tensor)
    wav = WV.mel2wav(mel)
    wav = wav[0].cpu().numpy()
    sf.write("music_waveglow.wav", wav, sr)

@timer
def test_with_config_and_waveglow(y_tensor, sr):
    print('\nTesting WaveGlowVocoder using config file and waveglow_path.')
    WV = WaveGlowVocoder(config_path="./load_test/config.json", waveglow_path="./load_test/waveglow_256channels_universal_v5.pt")
    mel = WV.wav2mel(y_tensor)
    wav = WV.mel2wav(mel)
    wav = wav[0].cpu().numpy()
    sf.write("music_waveglow_config_and_path.wav", wav, sr)

@timer
def test_with_config_waveglow_and_cpu(y_tensor, sr):
    print('\nTesting WaveGlowVocoder using config file, waveglow_path, and cpu.')
    WV = WaveGlowVocoder(config_path="./load_test/config.json", waveglow_path="./load_test/waveglow_256channels_universal_v5.pt", device="cpu")
    mel = WV.wav2mel(y_tensor.cpu())
    wav = WV.mel2wav(mel)
    wav = wav[0].cpu().numpy()
    sf.write("music_waveglow_cpu.wav", wav, sr)

@timer
def test_default_configuration_cpu(y_tensor, sr):
    print('\nTesting WaveGlowVocoder with default configuration on CPU.')
    WV = WaveGlowVocoder(device="cpu")
    mel = WV.wav2mel(y_tensor.cpu())
    wav = WV.mel2wav(mel)
    wav = wav[0].cpu().numpy()
    sf.write("music_waveglow_default_cpu.wav", wav, sr)

if __name__ == "__main__":
    y, sr = librosa.load(librosa.util.example(key='brahms'), sr=22050, mono=True, duration=10, offset=30)
    y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)
    sf.write("music_original.wav", y, sr)

    test_default_configuration(y_tensor, sr)
    test_with_config_file(y_tensor, sr)
    test_with_waveglow_path(y_tensor, sr)
    test_with_config_and_waveglow(y_tensor, sr)
    test_with_config_waveglow_and_cpu(y_tensor, sr)
    test_default_configuration_cpu(y_tensor, sr)
