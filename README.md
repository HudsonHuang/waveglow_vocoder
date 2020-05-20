# <h1 align="center">WaveGlow Vocoder</h1>
A vocoder that can convert audio to Mel-Spectrogram and reverse with [WaveGlow](https://github.com/NVIDIA/waveglow), all on GPU(if avaliable).  
Most code are extracted from [Tacotron2](https://github.com/NVIDIA/tacotron2/) and [WaveGlow](https://github.com/NVIDIA/waveglow) of Nvidia.
## <h2 align="center">Install</h1>
```
pip install waveglow-vocoder
```

## <h2 align="center">Example</h1>


![img](original.png)
![wav](original.wav)

![img](conveted.png)
![wav](conveted.wav)

## <h2 align="center">Performance</h1>
CPU(Intel i5):

GPU(GTX 1080Ti):

## <h2 align="center">Usage</h1>
Load wav file as usual
```
import librosa

y,sr = librosa.load(librosa.util.example_audio_file(), sr=22050, mono=True, duration=10, offset=30)
y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)
```
Apply mel transform, this would be done on GPU if avaliable.
```
from waveglow_vocoder import WaveGlowVocoder

WV = WaveGlowVocoder()
mel = WV.wav2mel(y_tensor)
```
Decoder it with Waveglow.  
>NOTE:  
 As the parameter of pre-trained model is alignment with [Tacotron2](https://github.com/NVIDIA/tacotron2/), one might get totally noise if the Mel spectrogram comes other function than *wav2mel*(an alias for TacotronSTFT.mel_spectrogram).  
 Support for librosa and torchaudio is under development.
```
wav = WV.mel2wav(mel)
```

## <h2 align="center">Other pretrained model / Train with your own data</h1>
This vocoder will download pre-trained model from [pytorch hub](https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/) on the first time of initialize.  
You can also download the latest model from [WaveGlow](https://github.com/NVIDIA/waveglow), or  with your own data and pass the path to the waveglow vocoder.

```
config_path = "your_config_of_model_training.json"
waveglow_path="your_model_path.pt"
WV = WaveGlowVocoder(waveglow_path=waveglow_path, config_path=config_path)
```
Then use it as usual.


## <h2 align="center">TODO</h1>
- pip
- WaveRNN Vocoder
- MelGAN Vocoder
- examples
- performance
- support librosa Mel input
- CPU support


## <h2 align="center">Reference</h1>
- [WaveGlow](https://github.com/NVIDIA/waveglow)
- [Tacotron2](https://github.com/NVIDIA/tacotron2/)
- [Wavenet Vocoder](https://github.com/r9y9/wavenet_vocoder)
