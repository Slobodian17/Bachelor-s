import numpy as np
import matplotlib.pyplot as plt
import wave
import math
import ctypes as ct
import time
from scipy.io import wavfile
from pesq import pesq


class FloatBits(ct.Structure):
    """C-compatible structure for float bit representation."""
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]


class Float(ct.Union):
    """Union for interpreting floats as bit fields."""
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]


def read_wav_file(filename, title):
    """Read and plot waveform data from WAV file."""
    with wave.open(filename, 'rb') as wav_file:
        params = wav_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = wav_file.readframes(nframes)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        time_data = np.arange(0, nframes) * (1.0 / framerate)

        plt.figure(figsize=(10, 4))
        plt.plot(time_data, wave_data, color='b', linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return wave_data, time_data


def next_power_of_2(x):
    """Return smallest power of 2 greater than or equal to |x|."""
    x = abs(x)
    if x == 0:
        return 0

    d = Float()
    d.value = x

    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1


def berouti(SNR):
    """Berouti noise suppression factor for power spectrum."""
    if -5.0 <= SNR <= 20.0:
        return 4 - (SNR * 3 / 20)
    elif SNR < -5.0:
        return 5
    return 1


def berouti1(SNR):
    """Berouti noise suppression factor for magnitude spectrum."""
    if -5.0 <= SNR <= 20.0:
        return 3 - (SNR * 2 / 20)
    elif SNR < -5.0:
        return 4
    return 1


def find_negative_indices(values):
    """Find indices where values are negative."""
    return [i for i, v in enumerate(values) if v < 0]


def spectral_subtraction_denoise(input_file, output_file):
    """Perform spectral subtraction noise reduction on input WAV file."""
    with wave.open(input_file, 'rb') as wav_file:
        params = wav_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        fs = framerate

        str_data = wav_file.readframes(nframes)
        x = np.frombuffer(str_data, dtype=np.int16)

    frame_len = 20 * fs // 1000
    overlap_percentage = 50
    len1 = frame_len * overlap_percentage // 100
    len2 = frame_len - len1

    threshold = 3
    exponent = 2.0
    beta = 0.002
    G = 0.9

    window = np.hamming(frame_len)
    window_gain = len2 / np.sum(window)

    nFFT = 2 * 2 ** next_power_of_2(frame_len)
    noise_mean = np.zeros(nFFT)

    # Initial noise estimate (first 5 frames)
    for i in range(5):
        noise_mean += np.abs(np.fft.fft(window * x[i * frame_len:(i + 1) * frame_len], nFFT))
    noise_mu = noise_mean / 5

    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    k = 0
    start_time = time.time()

    for n in range(Nframes):
        insign = window * x[k:k + frame_len]
        spec = np.fft.fft(insign, nFFT)
        magnitude = np.abs(spec)
        phase = np.angle(spec)

        SNRseg = 10 * np.log10(np.linalg.norm(magnitude, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        if exponent == 1.0:
            alpha = berouti1(SNRseg)
        else:
            alpha = berouti(SNRseg)

        sub_speech = np.power(magnitude, exponent) - alpha * np.power(noise_mu, exponent)
        diffw = sub_speech - beta * np.power(noise_mu, exponent)

        negative_indices = find_negative_indices(diffw)
        if negative_indices:
            sub_speech[negative_indices] = beta * np.power(noise_mu[negative_indices], exponent)

            if SNRseg < threshold:
                noise_temp = G * np.power(noise_mu, exponent) + (1 - G) * np.power(magnitude, exponent)
                noise_mu = np.power(noise_temp, 1 / exponent)

        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])

        x_phase = np.power(sub_speech, 1 / exponent) * (
                np.cos(phase) + 1j * np.sin(phase))

        xi = np.fft.ifft(x_phase).real
        xfinal[k:k + len2] = x_old + xi[:len1]
        x_old = xi[len1:frame_len]
        k += len2

    elapsed_time = time.time() - start_time
    print(f"Time taken for denoising: {elapsed_time:.2f} seconds")

    with wave.open(output_file, 'wb') as wf:
        wf.setparams(params)
        wave_data = (window_gain * xfinal).astype(np.int16)
        wf.writeframes(wave_data.tobytes())


def calculate_pesq(input_file, output_file):
    """Calculate PESQ score between input and output files."""
    with wave.open(input_file, 'rb') as orig_file, \
         wave.open(output_file, 'rb') as denoised_file:

        orig_data = np.frombuffer(orig_file.readframes(-1), dtype=np.int16)
        denoised_data = np.frombuffer(denoised_file.readframes(-1), dtype=np.int16)

    sampling_rate = 16000  # Modify if needed based on actual sampling rate
    score = pesq(sampling_rate, orig_data, denoised_data, 'wb')
    print(f"PESQ Score: {score:.3f}")


def main():
    input_file = "noisy_audio.wav"
    output_file = "result1.wav"

    spectral_subtraction_denoise(input_file, output_file)

    print("\nOriginal Noisy wave data:")
    read_wav_file(input_file, "Original Noisy Audio")

    print("\nSpectral Subtraction wave data:")
    read_wav_file(output_file, "Denoised Audio")

    calculate_pesq(input_file, output_file)


if __name__ == "__main__":
    main()
