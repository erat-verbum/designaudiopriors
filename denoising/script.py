#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:03:12 2020

@author: vnaray29
"""
import sys

sys.path.append("./")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import soundfile
from models.DDUnet.ddunet import DDUNet
import argparse
import os


def main(args):
    # Device configuration
    # Command to utilize either GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audo_filenames = [
        filename
        for filename in os.listdir(args.data_dir + "/")
        if os.path.isfile(os.path.join(args.data_dir, filename))
    ]

    for index, i in enumerate(sorted(audo_filenames)):

        noisy_audio = librosa.load(args.data_dir + "/" + i, 16000)[0]

        noisy_spec_numpy = librosa.stft(noisy_audio, n_fft=1023, hop_length=64)
        noisy_spec_real = np.expand_dims(noisy_spec_numpy.real, axis=0)
        noisy_spec_imag = np.expand_dims(noisy_spec_numpy.imag, axis=0)
        noisy_spec = np.concatenate((noisy_spec_real, noisy_spec_imag), axis=0)
        noisy_spec = np.expand_dims(noisy_spec, axis=0)
        print("Shape of Noisy Spectrogram", noisy_spec.shape)

        z = torch.FloatTensor(np.random.normal(0.0, 1.0, noisy_spec.shape)).to(device)
        noisy_spec = torch.FloatTensor(noisy_spec).to(device)

        if args.model_type == "ddunet":
            model = DDUNet(args).to(device)
            print("Model chosen {}".format(args.model_type))

        results_dir = os.path.join(args.results_dir)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )  # ADAM Optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        mse = nn.MSELoss()

        model.train()
        for iter_ in range(args.num_epochs):
            scheduler.step()

            enhanced_spec = model(z)
            loss = mse(enhanced_spec, noisy_spec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            enhanced_spec_numpy = (
                enhanced_spec.cpu().detach().numpy()[:, 0, :, :]
                + 1j * enhanced_spec.cpu().detach().numpy()[:, 1, :, :]
            )

        enhanced_audio = librosa.istft(
            enhanced_spec_numpy[0],
            win_length=1022,
            hop_length=64,
            length=len(noisy_audio),
        )

        soundfile.write(results_dir + "/" + i + ".wav", enhanced_audio, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ddunet", help="Model type")
    parser.add_argument(
        "--dilation_type", type=str, default="constant", help="Type of dilation"
    )
    parser.add_argument(
        "--num_input_channels", type=int, default=2, help="No of Input channels"
    )
    parser.add_argument(
        "--num_output_channels", type=int, default=2, help="No of output channels"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory of audio data"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="Results Directory"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=2000, help="number of epochs")

    main(parser.parse_args())
