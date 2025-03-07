import os
import librosa
import numpy as np
import soundfile
import IPython
import shutil
import subprocess
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2
from imutils import paths
from tqdm import tqdm

from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def download_and_extract_librispeech(destination_folder="/kaggle/working"):
    """
    Downloads and extracts the LibriSpeech dev-clean dataset.
    Moves all audio files into a dedicated clean speech folder.
    """
    librispeech_url = "http://openslr.elda.org/resources/12/dev-clean.tar.gz"
    librispeech_tar = os.path.join(destination_folder, "dev-clean.tar.gz")
    clean_speech_folder = os.path.join(destination_folder, "clean_speech")

    # Download dataset & extract files
    subprocess.run(["wget", librispeech_url, "-O", librispeech_tar])
    subprocess.run(["tar", "-xvf", librispeech_tar, "-C", destination_folder])

    os.makedirs(clean_speech_folder, exist_ok=True)

    # Move all .flac files to clean_speech
    for root, _, files in os.walk(os.path.join(destination_folder, "LibriSpeech/dev-clean")):
        for file in files:
            if file.endswith(".flac"):
                src = os.path.join(root, file)
                dest = os.path.join(clean_speech_folder, file)
                shutil.copy2(src, dest)

    # Clean up original dataset folder
    shutil.rmtree(os.path.join(destination_folder, "LibriSpeech"))

    print(f"Clean speech files saved to {clean_speech_folder}")


def download_and_extract_esc50(destination_folder="/kaggle/working"):
    """
    Downloads and extracts the ESC-50 dataset.
    Moves all noise audio files into a dedicated noise folder.
    """
    esc50_url = "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"
    esc50_zip = os.path.join(destination_folder, "ESC-50-master.zip")
    noise_folder = os.path.join(destination_folder, "noise")

    # Download dataset &dExtract files
    subprocess.run(["wget", esc50_url, "-O", esc50_zip])
    subprocess.run(["unzip", esc50_zip, "-d", destination_folder])

    # Create noise folder
    os.makedirs(noise_folder, exist_ok=True)

    # Move all .wav files to noise folder
    for root, _, files in os.walk(os.path.join(destination_folder, "ESC-50-master/audio")):
        for file in files:
            if file.endswith(".wav"):
                src = os.path.join(root, file)
                dest = os.path.join(noise_folder, file)
                shutil.copy2(src, dest)

    # Clean up original dataset folder
    shutil.rmtree(os.path.join(destination_folder, "ESC-50-master"))

    print(f"Noise files saved to {noise_folder}")


download_and_extract_librispeech()
download_and_extract_esc50()

count_clean = 0
duration_clean = 0
duration_noise = 0
sample_rate = 8000

for i in os.listdir('/kaggle/working/clean_speech'):
    count_clean += 1
    y, sr = librosa.load(os.path.join('/kaggle/working/clean_speech', i), sr=sample_rate)
    duration = librosa.get_duration(y=y, sr=sr)
    duration_clean += duration

print(f'Number of clean audio files are {count_clean}')
print(f'Total duration of all clean audios {duration_clean // 60}')

count_noise = 0
for i in os.listdir('/kaggle/working/noise'):
    count_noise += 1
    y, sr = librosa.load(os.path.join('/kaggle/working/noise', i), sr=sample_rate)
    duration = librosa.get_duration(y=y, sr=sr)
    duration_noise += duration

print(f'Number of Noisy audio files are {count_noise}')
print(f'Total duration of all noise audios {duration_noise // 60}')


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """This function take an audio and split into several frame
       in a numpy matrix of size (nb_frame,frame_length)"""

    sequence_sample_length = sound_data.shape[0]
    # Creating several audio frames using sliding windows
    sound_data_list = [sound_data[start:start + frame_length] for start in range(
        0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    # Combining all the frames to single matrix
    sound_data_array = np.vstack(sound_data_list)
    return sound_data_array


# Required variables for Audio
noise_dir = "/kaggle/working/noise/"
voice_dir = "/kaggle/working/clean_speech/"
path_save_spectrogram = "/kaggle/working/spectogram/"
sample_rate = 8000
min_duration = 1.0
frame_length = 8064
hop_length_frame = 8064
hop_length_frame_noise = 5000
nb_samples = 500
n_fft = 255
hop_length_fft = 63
dim_square_spec = int(n_fft / 2) + 1

clean_audio_files = os.listdir(voice_dir)
clean_random_audio = np.random.choice(clean_audio_files)
y, sr = librosa.load(os.path.join(voice_dir, clean_random_audio), sr=sample_rate)
clean = audio_to_audio_frame_stack(y, frame_length, hop_length_frame)
print("Clean Audio: {}".format(clean_random_audio))
print("Shape:{}".format(clean.shape))

noisy_audio_files = os.listdir(voice_dir)
noisy_random_audio = np.random.choice(clean_audio_files)
y, sr = librosa.load(os.path.join(voice_dir, noisy_random_audio), sr=sample_rate)
noise = audio_to_audio_frame_stack(y, frame_length, hop_length_frame)
print("Noise Audio: {}".format(noisy_random_audio))
print("Shape:{}".format(noise.shape))

clean = np.vstack(clean)
noise = np.vstack(noise)


def blend_noise_randomly(voice, noise, nb_samples, frame_length):
    """This function takes as input numpy arrays representing frames
    of voice sounds, noise sounds and the number of frames to be created
    and return numpy arrays with voice randomly blend with noise"""

    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = level_noise * noise[id_noise, :]
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy_voice


prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(voice=clean, noise=noise, nb_samples=10,
                                                                frame_length=frame_length)

samples_clean = []
samples_noisy_clean = []
for x in prod_voice:
    samples_clean.extend(x)

for x in prod_noisy_voice:
    samples_noisy_clean.extend(x)

import soundfile as sf

clean_nb_samples = prod_voice.shape[0]
# Save all frames in one file
clean_long = prod_voice.reshape(1, 10 * frame_length) * 10
# librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 1000)
sf.write("clean_long.wav", clean_long[0, :], 8000, 'PCM_24')

# from IPython.display import Audio
# Audio('clean_long.wav')

clean_nb_samples = prod_voice.shape[0]
# Save all frames in one file
clean_long = prod_voice.reshape(1, 10 * frame_length) * 10
# librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 1000)
sf.write("clean_long.wav", clean_long[0, :], 8000, 'PCM_24')

# from IPython.display import Audio
# Audio('noise_long.wav')


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    """This function takes an audio and convert into spectrogram,
       it returns the magnitude in dB and the phase"""

    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude_db, stftaudio_phase

def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""

    # we extract the magnitude vectors from the 256-point STFT vectors and
    # take the first 129-point by removing the symmetric half.

    nb_audio = numpy_audio.shape[0]
    # dim_square_spec = 256/2
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase


def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""

    list_sound_array = []

    count = 0
    for file in list_audio_files:
    # open the audio file
      try:
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        # Getting duration of audio file
        total_duration = librosa.get_duration(y=y, sr=sr)
      except ZeroDivisionError:
        count += 1

        # Check if the duration is atleast the minimum duration
      if (total_duration >= min_duration):
          list_sound_array.append(audio_to_audio_frame_stack(
              y, frame_length, hop_length_frame))
      else:
          print(
              f"The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)



#Data Prepare
def create_data(noise_dir, voice_dir,path_save_spectrogram, sample_rate,
min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):
    """This function will randomly blend some clean voices from voice_dir with some noises from noise_dir
    and save the spectrograms of noisy voice, noise and clean voices to disk as well as complex phase,
    time series and sounds. This aims at preparing datasets for denoising training. It takes as inputs
    parameters defined in args module"""

    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)

    def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')

        return lst

    list_noise_files = remove_ds_store(list_noise_files)
    list_voice_files = remove_ds_store(list_voice_files)

    nb_voice_files = len(list_voice_files)
    nb_noise_files = len(list_noise_files)


    # Extracting noise and voice from folder and convert to numpy
    noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
                                     frame_length, hop_length_frame_noise, min_duration)

    voice = audio_files_to_numpy(voice_dir, list_voice_files,
                                     sample_rate, frame_length, hop_length_frame, min_duration)

    # Blend some clean voices with random selected noises (and a random level of noise)
    prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
            voice, noise, nb_samples, frame_length)


    # Squared spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Create Amplitude and phase of the sounds
    m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            prod_voice, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
            prod_noise, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)

    np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
    np.save(path_save_spectrogram + 'noise_amp_db', m_amp_db_noise)             #Not required
    np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)
    print("List of spectrogram files:", os.listdir(path_save_spectrogram))
    for f in os.listdir(path_save_spectrogram):
        print(f, "Size:", os.path.getsize(os.path.join(path_save_spectrogram, f)))


if not os.path.exists('spectogram'):
    os.makedirs('spectogram')

noise_dir="/kaggle/working/noise/"
voice_dir="/kaggle/working/clean_speech/"
path_save_spectrogram="/kaggle/working/spectogram/"
sample_rate=8000
min_duration=1.0
frame_length=8064
hop_length_frame=8064
hop_length_frame_noise=5000
nb_samples=2000
n_fft=255
hop_length_fft=63

create_data(noise_dir=noise_dir,voice_dir=voice_dir,
            path_save_spectrogram=path_save_spectrogram,
            sample_rate=sample_rate,
            min_duration=min_duration,
            frame_length=frame_length,
            hop_length_frame=hop_length_frame,
            hop_length_frame_noise=hop_length_frame_noise,
            nb_samples=nb_samples,
            n_fft=n_fft,
            hop_length_fft=hop_length_fft)

noisy_voice_amp_db = np.load('/kaggle/working/spectogram/noisy_voice_amp_db.npy')
print(np.shape(noisy_voice_amp_db))


import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from IPython.display import Audio

# Define helper functions for audio processing

def load_spectrograms(path_save_spectrogram):
    noisy_spectrogram = np.load(os.path.join(path_save_spectrogram, 'noisy_voice_amp_db.npy'))
    clean_spectrogram = np.load(os.path.join(path_save_spectrogram, 'voice_amp_db.npy'))
    return noisy_spectrogram, clean_spectrogram

def spectrogram_to_waveform(spectrogram, n_fft, hop_length_fft):
    # Convert dB-scaled spectrogram back to amplitude
    amplitude = librosa.db_to_amplitude(spectrogram)
    # Pad amplitude to ensure the correct size for iSTFT
    if amplitude.shape[0] < n_fft:
        amplitude = np.pad(amplitude, ((0, n_fft - amplitude.shape[0]), (0, 0)), mode='constant')
    # Convert amplitude spectrogram to waveform
    waveform = librosa.istft(amplitude, hop_length=hop_length_fft, win_length=n_fft)
    return waveform

def plot_waveforms(samples_clean, samples_noisy, samples_denoised, sr=8000):
    plt.figure(figsize=(30, 10))

    plt.subplot(311)
    plt.title("Clean Speech")
    plt.plot(samples_clean)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(312)
    plt.title("Noisy Speech")
    plt.plot(samples_noisy)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(313)
    plt.title("Denoised Speech")
    plt.plot(samples_denoised)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# Define paths and parameters
path_save_spectrogram = "/kaggle/working/spectogram/"
n_fft = 255
hop_length_fft = 63
sample_rate = 8000

# Load spectrograms
noisy_spectrogram, clean_spectrogram = load_spectrograms(path_save_spectrogram)

# Assume the model is already trained and you have the denoised spectrograms
# For demonstration, let's assume denoised_spectrogram is the same as clean_spectrogram
denoised_spectrogram = clean_spectrogram  # Replace this with the actual denoised output from your model

# Convert spectrograms to waveforms
noisy_waveform = spectrogram_to_waveform(noisy_spectrogram[0], n_fft, hop_length_fft)
clean_waveform = spectrogram_to_waveform(clean_spectrogram[0], n_fft, hop_length_fft)
denoised_waveform = spectrogram_to_waveform(denoised_spectrogram[0], n_fft, hop_length_fft)

# Save waveforms as audio files
sf.write("noisy.wav", noisy_waveform, sample_rate)
sf.write("clean.wav", clean_waveform, sample_rate)
sf.write("denoised.wav", denoised_waveform, sample_rate)

# Plot waveforms
plot_waveforms(clean_waveform, noisy_waveform, denoised_waveform, sr=sample_rate)
#
# # Play audio files
# display(Audio("noisy.wav"))
# display(Audio("clean.wav"))
# display(Audio("denoised.wav"))
#
#

noisy_voice = np.load("/kaggle/working/spectogram/noisy_voice_amp_db.npy")
voice = np.load("/kaggle/working/spectogram/voice_amp_db.npy")
noise = noisy_voice-voice

def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec
def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec -6 )/82
    return matrix_spec
from scipy import stats


X_in = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
X_ou = np.load(path_save_spectrogram +'voice_amp_db'+".npy")
#Model of noise to predict
X_ou = X_in - X_ou

#Check distribution
print(stats.describe(X_in.reshape(-1,1)))
print(stats.describe(X_ou.reshape(-1,1)))

#to scale between -1 and 1
X_in = scaled_in(X_in)
X_ou = scaled_ou(X_ou)

#Check shape of spectrograms
print(X_in.shape)
print(X_ou.shape)
#Check new distribution
print(stats.describe(X_in.reshape(-1,1)))
print(stats.describe(X_ou.reshape(-1,1)))


#Reshape for training
X_in = X_in[:,:,:]
X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
X_ou = X_ou[:,:,:]
X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)
# print(X_in.shape)
# print(X_out.shape)

X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.20, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l1_l2

def unet(input_size=(128, 128, 1)):
    size_filter_in = 16
    kernel_init = 'he_normal'
    activation_layer = LeakyReLU()

    regularizer = l1_l2(l1=1e-5, l2=1e-4)

    inputs = Input(input_size)
    conv_args = dict(padding='same',
                     kernel_initializer=kernel_init,
                     activation=activation_layer,
                     kernel_regularizer=regularizer)

    # Encoder
    conv1 = Conv2D(size_filter_in, 3, **conv_args)(inputs)
    conv1 = Conv2D(size_filter_in, 3, **conv_args)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(size_filter_in*2, 3, **conv_args)(pool1)
    conv2 = Conv2D(size_filter_in*2, 3, **conv_args)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(size_filter_in*4, 3, **conv_args)(pool2)
    conv3 = Conv2D(size_filter_in*4, 3, **conv_args)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(size_filter_in*8, 3, **conv_args)(pool3)
    conv4 = Conv2D(size_filter_in*8, 3, **conv_args)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(size_filter_in*16, 3, **conv_args)(pool4)
    conv5 = Conv2D(size_filter_in*16, 3, **conv_args)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2D(size_filter_in*8, 2, **conv_args)(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(size_filter_in*8, 3, **conv_args)(merge6)
    conv6 = Conv2D(size_filter_in*8, 3, **conv_args)(conv6)

    up7 = Conv2D(size_filter_in*4, 2, **conv_args)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(size_filter_in*4, 3, **conv_args)(merge7)
    conv7 = Conv2D(size_filter_in*4, 3, **conv_args)(conv7)

    up8 = Conv2D(size_filter_in*2, 2, **conv_args)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(size_filter_in*2, 3, **conv_args)(merge8)
    conv8 = Conv2D(size_filter_in*2, 3, **conv_args)(conv8)

    up9 = Conv2D(size_filter_in, 2, **conv_args)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(size_filter_in, 3, **conv_args)(merge9)
    conv9 = Conv2D(size_filter_in, 3, **conv_args)(conv9)

    # Output
    conv10 = Conv2D(1, 1, activation='tanh')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

    return model



import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy import stats
import matplotlib.pyplot as plt

def psnr(y_true, y_pred):
    max_pixel = 1.0
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)
# def psnr(y_true, y_pred, max_pixel_value=1.0):
#     """
#     Compute Peak Signal-to-Noise Ratio (PSNR) between two spectrograms (or images).
#
#     Parameters:
#         y_true (np.array): Ground truth spectrogram (shape: H x W).
#         y_pred (np.array): Denoised/predicted spectrogram (shape: H x W).
#         max_pixel_value (float): Maximum possible pixel value (1.0 if normalized to [0, 1]).
#
#     Returns:
#         float: PSNR value in decibels (dB).
#     """
#     mse = np.mean((y_true - y_pred) ** 2)
#     if mse == 0:
#         return np.inf  # Perfect reconstruction
#     return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def plot_waveforms(samples_clean, samples_noisy, samples_denoised, sr=8000):
    plt.figure(figsize=(30, 10))

    plt.subplot(311)
    plt.title("Clean Speech")
    plt.plot(samples_clean)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(312)
    plt.title("Noisy Speech")
    plt.plot(samples_noisy)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(313)
    plt.title("Denoised Speech")
    plt.plot(samples_denoised)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def plot_spectrograms(generator_nn, X_test, y_test, num_samples=3):
    # Randomly select samples
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    noisy_samples = X_test[indices]
    denoised_samples = generator_nn.predict(noisy_samples)

    plt.figure(figsize=(15, 3*num_samples))
    for i in range(num_samples):
        # Compute spectrograms
        noisy_spec = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_samples[i].squeeze())), ref=np.max)
        denoised_spec = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_samples[i].squeeze())), ref=np.max)

        # Plot spectrograms
        plt.subplot(num_samples, 2, 2*i + 1)
        librosa.display.specshow(noisy_spec.squeeze(), sr=44100, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Noisy Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        plt.subplot(num_samples, 2, 2*i + 2)
        librosa.display.specshow(denoised_spec.squeeze(), sr=44100, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Denoised Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()



import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def training_unet(path_save_spectrogram, weights_path, epochs, batch_size):
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")

    # Use MirroredStrategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()

    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Load data
    X_in = np.load(path_save_spectrogram + 'noisy_voice_amp_db.npy')
    X_ou = np.load(path_save_spectrogram + 'voice_amp_db.npy')
    X_ou = X_in - X_ou

    # Scale data
    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    # Reshape for training
    X_in = X_in[:, :, :]
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    X_ou = X_ou[:, :, :]
    X_ou = X_ou.reshape(X_ou.shape[0], X_ou.shape[1], X_ou.shape[2], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.30, random_state=42)

    # Create the model inside the strategy scope
    with strategy.scope():
        generator_nn = unet()

        # Save best model checkpoint
        checkpoint = ModelCheckpoint(
            weights_path + '/model_unet_best.keras',
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            mode='auto'
        )

        # Compile the model
        generator_nn.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=[psnr, ssim, 'mae']
        )

    # Training
    history = generator_nn.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size * strategy.num_replicas_in_sync,  # Scale batch size
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1,
        validation_data=(X_test, y_test)
    )

    # Plot training and validation metrics
    plot_training_metrics(history)

    # You can add additional visualization or denoising evaluation if needed

def plot_training_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    psnr_values = history.history['psnr']
    val_psnr_values = history.history['val_psnr']
    ssim_values = history.history['ssim']
    val_ssim_values = history.history['val_ssim']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, mae, 'r', label='Training MAE')
    plt.plot(epochs, val_mae, 'b', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, psnr_values, 'r', label='Training PSNR')
    plt.plot(epochs, val_psnr_values, 'b', label='Validation PSNR')
    plt.title('Training and validation PSNR')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, ssim_values, 'r', label='Training SSIM')
    plt.plot(epochs, val_ssim_values, 'b', label='Validation SSIM')
    plt.title('Training and validation SSIM')
    plt.legend()

    plt.tight_layout()
    plt.show()



import time

if not os.path.exists('weights'):
    os.makedirs('weights')

start_time = time.time()

training_unet(path_save_spectrogram, './weights', epochs=  100 ,batch_size=64)

end_time = time.time()
elapsed_time = end_time - start_time
print("Time taken for denoising:", elapsed_time, "seconds")