""" WIENER FILTER """

# import numpy as np
# import soundfile as sf
# import pesq
# import resampy
# import matplotlib.pyplot as plt
#
# def wiener_filter_denoise(signal, noise, snr):
#     """
#     Apply Wiener filter for audio denoising.
#
#     Parameters:
#         signal (ndarray): Clean signal.
#         noise (ndarray): Noisy signal.
#         snr (float): Signal-to-noise ratio.
#
#     Returns:
#         ndarray: Denoised signal.
#     """
#     # Estimate power spectral densities
#     signal_power = np.abs(np.fft.fft(signal))**2
#     noise_power = np.abs(np.fft.fft(noise))**2
#
#     # Calculate Wiener filter coefficients
#     wiener_filter = 1 / (1 + (noise_power / signal_power) * (10**(-snr/10)))
#
#     # Apply Wiener filter
#     denoised_signal_freq = np.fft.fft(signal) * wiener_filter
#     denoised_signal = np.fft.ifft(denoised_signal_freq)
#
#     return np.real(denoised_signal)
#
# # Read the noisy audio file
# noisy_audio, fs = sf.read('noisy_audio5.wav')
#
# # Read the clean audio file
# clean_audio, _ = sf.read('clean_audio5.wav')
#
# # Resample both audio files if necessary
# target_fs = 16000  # Target sampling frequency
# if fs != target_fs:
#     noisy_audio = resampy.resample(noisy_audio, fs, target_fs)
#     clean_audio = resampy.resample(clean_audio, fs, target_fs)
#     fs = target_fs
#
# # Generate a noise reference (could be estimated from a silent part of the audio)
# noise_reference = np.random.normal(0, 0.5, len(noisy_audio))
#
# # Set SNR (Signal-to-Noise Ratio)
# snr = 15
#
# # Apply Wiener filter for denoising
# denoised_audio = wiener_filter_denoise(noisy_audio, noise_reference, snr)
# # Save the denoised audio
# sf.write('denoised_audio5.wav', denoised_audio, fs)
# # Calculate PESQ score
# pesq_score = pesq.pesq(fs, clean_audio, denoised_audio, 'wb')
# print("PESQ Score:", pesq_score)
#
# # Plot clean data, noisy data, and denoised data
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.plot(clean_audio)
# plt.title('Clean Data')
#
# plt.subplot(3, 1, 2)
# plt.plot(noisy_audio)
# plt.title('Noisy Data')
#
# plt.subplot(3, 1, 3)
# plt.plot(denoised_audio)
# plt.title('Denoised Data')
#
# plt.tight_layout()
# plt.show()
# #
#



# ############## CODE FOR NOISE AND SPEECH AUDIO MIXING #########################



import numpy as np
import soundfile as sf
import resampy

def mix_speech_with_noise(speech_file, noise_file, snr):
    """
    Mix speech with noise at a specified SNR.

    Parameters:
        speech_file (str): Path to the speech file.
        noise_file (str): Path to the noise file.
        snr (float): Signal-to-noise ratio in dB.

    Returns:
        ndarray: Noisy speech.
        ndarray: Clean speech.
        int: Sampling rate of the mixed speech.
    """
    # Read speech and noise files
    speech, fs_speech = sf.read(speech_file)
    noise, fs_noise = sf.read(noise_file)

    # Resample the audio files if the sampling rates are different
    if fs_speech != fs_noise:
        # Resample the audio with lower sampling rate to match the higher one
        if fs_speech < fs_noise:
            speech = resampy.resample(speech, fs_speech, fs_noise)
            fs_speech = fs_noise
        else:
            noise = resampy.resample(noise, fs_noise, fs_speech)
            fs_noise = fs_speech

    # Make sure both have the same length
    min_len = min(len(speech), len(noise))
    speech = speech[:min_len]
    noise = noise[:min_len]

    # Adjust the noise level to achieve the desired SNR
    noise_rms = np.sqrt(np.mean(noise**2))
    speech_rms = np.sqrt(np.mean(speech**2))
    scale_factor = 10**(-snr / 20) * speech_rms / noise_rms
    scaled_noise = noise * scale_factor

    # Mix speech with noise
    noisy_speech = speech + scaled_noise

    return noisy_speech, speech, fs_speech

# Paths to speech and noise files
speech_file = 'sound5.flac'
noise_file = 'sound5.wav'

# Set desired SNR (Signal-to-Noise Ratio)
snr = 10  # in dB

# Create noisy audio
noisy_audio, clean_audio, fs_speech = mix_speech_with_noise(speech_file, noise_file, snr)

# Save the noisy audio and the clean speech
sf.write('noisy_audio5.wav', noisy_audio, fs_speech)
sf.write('clean_audio5.wav', clean_audio, fs_speech)
