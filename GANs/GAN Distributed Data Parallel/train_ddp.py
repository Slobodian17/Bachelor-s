import os
import random
import shutil
import subprocess
import warnings
import time
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pesq import pesq
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm

import IPython
import json
import argparse
import math
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def download_and_extract_librispeech(destination_folder="/kaggle/working"):
    librispeech_url = "http://openslr.elda.org/resources/12/dev-clean.tar.gz"
    librispeech_tar = os.path.join(destination_folder, "dev-clean.tar.gz")
    clean_speech_folder = os.path.join(destination_folder, "clean_speech")

    subprocess.run(["wget", librispeech_url, "-O", librispeech_tar])
    subprocess.run(["tar", "-xvf", librispeech_tar, "-C", destination_folder])

    os.makedirs(clean_speech_folder, exist_ok=True)

    for root, _, files in os.walk(os.path.join(destination_folder, "LibriSpeech/dev-clean")):
        for file in files:
            if file.endswith(".flac"):
                src = os.path.join(root, file)
                dest = os.path.join(clean_speech_folder, file)
                shutil.copy2(src, dest)

    shutil.rmtree(os.path.join(destination_folder, "LibriSpeech"))

    print(f"Clean speech files saved to {clean_speech_folder}")


def download_and_extract_esc50(destination_folder="/kaggle/working"):
    esc50_url = "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"
    esc50_zip = os.path.join(destination_folder, "ESC-50-master.zip")
    noise_folder = os.path.join(destination_folder, "noise")

    subprocess.run(["wget", esc50_url, "-O", esc50_zip])
    subprocess.run(["unzip", esc50_zip, "-d", destination_folder])

    os.makedirs(noise_folder, exist_ok=True)

    for root, _, files in os.walk(os.path.join(destination_folder, "ESC-50-master/audio")):
        for file in files:
            if file.endswith(".wav"):
                src = os.path.join(root, file)
                dest = os.path.join(noise_folder, file)
                shutil.copy2(src, dest)

    shutil.rmtree(os.path.join(destination_folder, "ESC-50-master"))

    print(f"Noise files saved to {noise_folder}")


def list_audio_files(folder, exts=(".wav", ".flac")):
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)]


def frame_audio(y, frame_length, hop_length):
    n_frames = 1 + (len(y) - frame_length) // hop_length
    return np.stack([
        y[i * hop_length: i * hop_length + frame_length]
        for i in range(n_frames)
    ], axis=0)


def compute_mag_spectrogram(frame, n_fft, hop_length):
    S = librosa.stft(frame, n_fft=n_fft, hop_length=hop_length, center=False)
    return np.abs(S)


def pad_to_multiple(spec: np.ndarray, mult=8) -> np.ndarray:
    F, T = spec.shape
    pad_F = (mult - (F % mult)) % mult
    pad_T = (mult - (T % mult)) % mult
    if pad_F or pad_T:
        spec = np.pad(
            spec,
            ((0, pad_F), (0, pad_T)),
            mode="reflect"
        )
    return spec


class AudioDenoiseDataset(Dataset):
    def __init__(self,
                 clean_folder,
                 noise_folder,
                 sample_rate=8000,
                 frame_length=8064,
                 hop_length_frame=8064,
                 n_fft=255,
                 hop_length_fft=63,
                 nb_samples=1000,
                 noise_levels=(0.2, 0.8)):
        super().__init__()
        self.clean_files = list_audio_files(clean_folder)
        self.noise_files = list_audio_files(noise_folder)
        self.sr = sample_rate
        self.frame_len = frame_length
        self.hop_len_frame = hop_length_frame
        self.n_fft = n_fft
        self.hop_fft = hop_length_fft
        self.nb_samples = nb_samples
        self.noise_levels = noise_levels

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        clean_path = random.choice(self.clean_files)
        noise_path = random.choice(self.noise_files)

        clean_y, _ = librosa.load(clean_path, sr=self.sr)
        noise_y, _ = librosa.load(noise_path, sr=self.sr)

        clean_frames = frame_audio(clean_y, self.frame_len, self.hop_len_frame)
        noise_frames = frame_audio(noise_y, self.frame_len, self.hop_len_frame)

        c = clean_frames[np.random.randint(len(clean_frames))]
        n = noise_frames[np.random.randint(len(noise_frames))]

        level = np.random.uniform(*self.noise_levels)
        noisy = c + level * n

        spec_c = compute_mag_spectrogram(c, self.n_fft, self.hop_fft)
        spec_n = compute_mag_spectrogram(noisy, self.n_fft, self.hop_fft)

        def norm(x):
            x = np.log1p(x)
            x = x / np.max(np.abs(x))
            return x.astype(np.float32)

        spec_c = norm(spec_c)
        spec_n = norm(spec_n)

        def pad_to_multiple(spec, mult=8):
            F, T = spec.shape
            pad_F = (mult - (F % mult)) % mult
            pad_T = (mult - (T % mult)) % mult
            if pad_F or pad_T:
                spec = np.pad(spec,
                              ((0, pad_F), (0, pad_T)),
                              mode="reflect")
            return spec

        spec_c = pad_to_multiple(spec_c, mult=8)
        spec_n = pad_to_multiple(spec_n, mult=8)

        return (
            torch.from_numpy(spec_n)[None, ...],
            torch.from_numpy(spec_c)[None, ...],
            noisy.astype(np.float32),
            c.astype(np.float32),
        )

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batch, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        return self.gamma * out + x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class TFDenseUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, growth_rate=16, num_layers=4):
        super().__init__()
        self.enc1 = DenseBlock(in_channels, growth_rate, num_layers)
        enc1_out = in_channels + num_layers * growth_rate
        self.down1 = nn.Conv2d(enc1_out, base_channels, kernel_size=4, stride=2, padding=1)

        self.enc2 = DenseBlock(base_channels, growth_rate, num_layers)
        enc2_out = base_channels + num_layers * growth_rate
        self.down2 = nn.Conv2d(enc2_out, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.enc3 = DenseBlock(base_channels * 2, growth_rate, num_layers)
        enc3_out = base_channels * 2 + num_layers * growth_rate
        self.down3 = nn.Conv2d(enc3_out, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.bottleneck = DenseBlock(base_channels * 4, growth_rate, num_layers)
        bottleneck_out = base_channels * 4 + num_layers * growth_rate
        self.attn = SelfAttention(bottleneck_out)

        self.up3 = nn.ConvTranspose2d(bottleneck_out, base_channels * 2, kernel_size=4, stride=2, padding=1)
        dec3_in = base_channels * 2 + enc3_out
        self.dec3 = DenseBlock(dec3_in, growth_rate, num_layers)
        dec3_out = dec3_in + num_layers * growth_rate

        self.up2 = nn.ConvTranspose2d(dec3_out, base_channels, kernel_size=4, stride=2, padding=1)
        dec2_in = base_channels + enc2_out
        self.dec2 = DenseBlock(dec2_in, growth_rate, num_layers)
        dec2_out = dec2_in + num_layers * growth_rate

        self.up1 = nn.ConvTranspose2d(dec2_out, base_channels, kernel_size=4, stride=2, padding=1)
        dec1_in = base_channels + enc1_out
        self.dec1 = DenseBlock(dec1_in, growth_rate, num_layers)
        dec1_out = dec1_in + num_layers * growth_rate

        self.final_conv = nn.Conv2d(dec1_out, in_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        b = self.bottleneck(d3)
        b = self.attn(b)

        u3 = self.up3(b)
        cat3 = torch.cat([u3, e3], dim=1)
        dec3 = self.dec3(cat3)

        u2 = self.up2(dec3)
        cat2 = torch.cat([u2, e2], dim=1)
        dec2 = self.dec2(cat2)

        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, e1], dim=1)
        dec1 = self.dec1(cat1)

        out = self.final_conv(dec1)
        return out


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        ch = base_channels
        for i in range(1, 5):
            layers.append(nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(ch * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch *= 2
        layers.append(nn.Conv2d(ch, 1, kernel_size=4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator() for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        results = []
        for D in self.discriminators:
            results.append(D(x))
            x = self.downsample(x)
        return results


CONFIG = {
    "sample_rate": 8000,
    "frame_length": 8064,
    "hop_length_frame": 8000,
    "n_fft": 255,
    "hop_length_fft": 63,
    "batch_size_per_gpu": 8,
    "lr": 1e-4,
    "noise_levels": (0.2, 0.8),
    "checkpoint_dir": "/kaggle/working/checkpoints"
}


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30)
    )
    torch.cuda.set_device(rank)
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'


def cleanup():
    dist.destroy_process_group()


def calculate_pesq(clean, denoised, sr):
    try:
        return pesq(sr, clean, denoised, 'nb')
    except Exception as e:
        print(f"PESQ failed: {str(e)}")
        return None


def ddp_train(rank, world_size, train_dataset, val_dataset, num_epochs):
    setup(rank, world_size)

    generator = TFDenseUNetGenerator().to(rank)
    discriminator = MultiScaleDiscriminator().to(rank)
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    g_opt = torch.optim.Adam(generator.parameters(), lr=CONFIG["lr"])
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr"])
    l1_loss = nn.L1Loss().to(rank)

    if rank == 0:
        history = {
            "train_mae": [], "train_psnr": [], "train_ssim": [],
            "val_mae": [], "val_psnr": [], "val_ssim": [], "val_pesq": []
        }
        best_mae = float('inf')
    else:
        history = None
        best_mae = None

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size_per_gpu"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    for epoch in range(num_epochs):
        generator.train()
        train_sampler.set_epoch(epoch)

        total_mae, total_psnr, total_ssim = 0.0, 0.0, 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=rank != 0)

        for noisy_spec, clean_spec, *_ in progress:
            noisy_spec = noisy_spec.to(rank, non_blocking=True)
            clean_spec = clean_spec.to(rank, non_blocking=True)

            fake_clean = generator(noisy_spec)
            loss = l1_loss(fake_clean, clean_spec)

            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

            with torch.no_grad():
                fake_np = fake_clean.detach().cpu().numpy()
                clean_np = clean_spec.detach().cpu().numpy()

                batch_mae = loss.item() * noisy_spec.size(0)
                batch_psnr = 0.0
                batch_ssim = 0.0

                for i in range(fake_np.shape[0]):
                    f = fake_np[i, 0]
                    c = clean_np[i, 0]
                    data_range = f.max() - f.min()
                    batch_psnr += psnr(c, f, data_range=data_range)
                    batch_ssim += ssim(c, f, data_range=data_range)

                total_mae += batch_mae
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_samples += noisy_spec.size(0)

            progress.set_postfix({"Loss": total_mae / total_samples})

        total_mae_tensor = torch.tensor(total_mae).to(rank)
        total_psnr_tensor = torch.tensor(total_psnr).to(rank)
        total_ssim_tensor = torch.tensor(total_ssim).to(rank)
        total_samples_tensor = torch.tensor(total_samples).to(rank)

        dist.all_reduce(total_mae_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ssim_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_mae = total_mae_tensor.item() / total_samples_tensor.item()
            avg_psnr = total_psnr_tensor.item() / total_samples_tensor.item()
            avg_ssim = total_ssim_tensor.item() / total_samples_tensor.item()

            history["train_mae"].append(avg_mae)
            history["train_psnr"].append(avg_psnr)
            history["train_ssim"].append(avg_ssim)

        if rank == 0:
            val_metrics = validate_and_save(
                generator.module,
                val_dataset,
                epoch,
                history,
                best_mae
            )
            best_mae = min(best_mae, val_metrics["mae"])

    cleanup()


def validate_and_save(model, val_dataset, epoch, history, best_mae):
    model.eval()
    metrics = {
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "pesq": 0.0
    }
    valid_samples = 0
    pesq_results = []
    wav_pairs = []

    with torch.no_grad():
        loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
        for noisy_spec, clean_spec, noisy_wav, clean_wav in loader:
            try:
                den_spec = model(noisy_spec.cuda())

                mae = F.l1_loss(den_spec, clean_spec.cuda()).item()
                den_np = den_spec.squeeze().cpu().numpy()
                clean_np = clean_spec.squeeze().cpu().numpy()

                data_range = den_np.max() - den_np.min()
                psnr_val = psnr(clean_np, den_np, data_range=data_range)
                ssim_val = ssim(clean_np, den_np, data_range=data_range)

                metrics["mae"] += mae * noisy_spec.size(0)
                metrics["psnr"] += psnr_val * noisy_spec.size(0)
                metrics["ssim"] += ssim_val * noisy_spec.size(0)
                valid_samples += noisy_spec.size(0)

                for i in range(noisy_spec.size(0)):
                    frame_length = CONFIG["frame_length"]
                    n_fft = CONFIG["n_fft"]
                    hop_length_fft = CONFIG["hop_length_fft"]

                    T_original = (frame_length - n_fft) // hop_length_fft + 1
                    den_spec_trimmed = den_np[i][:, :T_original]

                    S_noisy = librosa.stft(
                        noisy_wav[i].numpy(),
                        n_fft=n_fft,
                        hop_length=hop_length_fft,
                        center=False
                    )
                    phase = np.angle(S_noisy)

                    wav_denoised = librosa.istft(
                        den_spec_trimmed * np.exp(1j * phase),
                        hop_length=hop_length_fft,
                        center=False
                    )

                    clean_audio = clean_wav[i].numpy()
                    min_len = min(len(clean_audio), len(wav_denoised))
                    if min_len < 160:
                        continue

                    clean_audio = clean_audio[:min_len]
                    wav_denoised = wav_denoised[:min_len]

                    clean_audio = clean_audio / (np.max(np.abs(clean_audio)) + 1e-9)
                    wav_denoised = wav_denoised / (np.max(np.abs(wav_denoised)) + 1e-9)

                    wav_pairs.append((clean_audio, wav_denoised))

            except Exception as e:
                print(f"[Validation] Error processing batch: {str(e)}")
                continue

    if wav_pairs:
        with mp.Pool(processes=4) as pool:
            pesq_results = pool.starmap(calculate_pesq,
                                        [(clean, denoised, CONFIG["sample_rate"])
                                         for clean, denoised in wav_pairs])

        metrics["pesq"] = np.nanmean([x for x in pesq_results if x is not None])
        valid_pesq_samples = len([x for x in pesq_results if x is not None])
    else:
        metrics["pesq"] = 0.0
        valid_pesq_samples = 0

    if valid_samples > 0:
        metrics["mae"] /= valid_samples
        metrics["psnr"] /= valid_samples
        metrics["ssim"] /= valid_samples

    if valid_pesq_samples == 0:
        metrics["pesq"] = 0.0

    if valid_samples > 0:
        history["val_mae"].append(metrics["mae"])
        history["val_psnr"].append(metrics["psnr"])
        history["val_ssim"].append(metrics["ssim"])
        history["val_pesq"].append(metrics["pesq"])

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    try:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "metrics": metrics,
            "best_mae": best_mae,
            "history": history
        }, os.path.join(CONFIG["checkpoint_dir"], f"ckpt_epoch{epoch}.pth"))

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "best_mae": best_mae,
                "history": history
            }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))

        with open(os.path.join(CONFIG["checkpoint_dir"], "history.json"), "w") as f:
            json.dump(history, f)

    except Exception as e:
        print(f"Checkpoint save failed: {str(e)}")

    print(f"[Epoch {epoch + 1}] Val MAE: {metrics['mae']:.4f} | "
          f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.3f} | "
          f"PESQ: {metrics['pesq']:.3f} (from {valid_pesq_samples} samples)")

    return metrics


def main(args):
    if args.download:
        download_and_extract_librispeech()
        download_and_extract_esc50()

    clean_dir = "/kaggle/working/clean_speech"
    noise_dir = "/kaggle/working/noise"

    dataset = AudioDenoiseDataset(
        clean_folder=clean_dir,
        noise_folder=noise_dir,
        sample_rate=8000,
        frame_length=8064,
        hop_length_frame=8000,
        n_fft=255,
        hop_length_fft=63,
        nb_samples=args.nb_samples,
        noise_levels=(0.2, 0.8),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    noisy_batch, clean_batch, noisy_wave_batch, clean_wave_batch = next(iter(loader))
    print("Noisy spectrogram batch:", noisy_batch.shape)
    print("Clean spectrogram batch:", clean_batch.shape)
    full = AudioDenoiseDataset(
        clean_folder="/kaggle/working/clean_speech",
        noise_folder="/kaggle/working/noise",
        nb_samples=args.nb_samples,
    )
    n_train = int(0.8 * len(full))
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, len(full) - n_train])

    world_size = torch.cuda.device_count()

    start_time = time.time()

    mp.spawn(
        ddp_train,
        args=(world_size, train_ds, val_ds, args.epochs),
        nprocs=world_size,
        join=True
    )

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--nb_samples", type=int, default=5000,
                        help="Number of training samples to generate")
    parser.add_argument("--download", type=int, choices=[0, 1], default=1,
                        help="Download datasets (1) or use existing (0)")
    args = parser.parse_args()

    main(args)
