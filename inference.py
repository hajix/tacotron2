import numpy as np
import soundfile as sf
import sys
import time

import torch

sys.path.append('waveglow/')
from hparams import CreateHParams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


class TTS:
    def __init__(
            self,
            tacotron_model_path,
            waveglow_model_path,
            hparams,
            device
    ):
        self.hparams = hparams
        self.device = device
        # tacotron 2
        self.tacotron = load_model(self.hparams)
        tacotron_state_dict = torch.load(tacotron_model_path)['state_dict']
        self.tacotron.load_state_dict(tacotron_state_dict)
        self.tacotron.to(self.device).eval()
        # waveglow
        self.waveglow = torch.load(waveglow_model_path)['model']
        self.waveglow.to(self.device).eval()
        self.denoiser = Denoiser(self.waveglow)

    def text_to_speech(self, text, audio_save_path):
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))
        sequence = torch.from_numpy(sequence).to(self.device)
        sequence = sequence.unsqueeze(0).long()

        with torch.no_grad():
            output = self.tacotron.inference(sequence)
            audio = self.waveglow.infer(output[1], sigma=0.85)
        audio = audio.cpu().numpy()[0]

        sf.write(audio_save_path, audio, self.hparams.sampling_rate)
        return audio


if __name__ == '__main__':
    tacotron_model_path = 'resources/tacotron2_statedict.pt'
    waveglow_model_path = 'resources/waveglow_256channels_universal_v5.pt'
    hparams = CreateHParams()
    device = torch.device('cuda:0')

    tts = TTS(
        tacotron_model_path=tacotron_model_path,
        waveglow_model_path=waveglow_model_path,
        hparams=hparams,
        device=device
    )
    t0 = time.time()
    audio = tts.text_to_speech(
        'I make some love with reza akbar. His family is akbar.',
        'tmp.wav'
    )
    delta_t = time.time() - t0
    audio_len = audio.shape[0] / hparams.sampling_rate
    print(f'rtf = {audio_len}/{delta_t} = {round(audio_len/delta_t, 5)}')
