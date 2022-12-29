# Importing libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# signal processing and feature extraction libraries:
import librosa
import librosa.display
import IPython.display as ipd
import scipy.fft
from scipy.signal import welch
from PyEMD import EMD
from skimage.restoration import denoise_wavelet
from scipy.signal import filtfilt
import scipy

import warnings
warnings.filterwarnings('ignore')

# Variables:
signal = []  # original signal
sampling_rate = 0  # samplying frequency
imfs = []  # imfs signals
wl_signal = []  # wavelet-denoising signal
filtered_signal = []  # digital-filter signal
emd_wavelet_signal = []  # signal after using EMD and Wavelet-Denoising methods
emd_dfilter_signal = []  # signal after using EMD and Digital-Filter methods
emd_wl_dfilter_signal = []  # signal after using EMD, Wavelet-Denoising and Digital-Filter methods


# 1. VISUALIZATION FUNCTIONS:
# Function to display the audio file based on its index:
def display_audio(file_path, audio_index):
    return ipd.Audio(file_path[audio_index])


# Function to extract the signal and sampling frequency:
def signal_extraction(file_path, audio_index, sr=''):
    global signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
    print(f'Signal: {signal[:10]}')
    print(f'Signal Shape: {signal.shape}')
    print(f'Sample Rate: {sr}')


# Function to visualize the signal in Time-Domain:
def signal_time_domain(file_path, audio_index, sr='', zoom_1='', zoom_2=''):
    global signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)

    zoom_1 = 1 if zoom_1 == 0 else zoom_1
    plt.figure(figsize=(15, 5))
    if zoom_1 and zoom_2:
        librosa.display.waveshow(signal[zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
        plt.title(f'Zoomed Row Audio Signal ({zoom_1} - {zoom_2})')
    else:
        librosa.display.waveshow(signal, sr=sampling_rate, alpha=0.6)
        plt.title(f'Complete Row Audio Signal')
    plt.ylim((-0.3, 0.3))
    return plt.show()


# Function to display the signal in Frequency-Domain (Spectrum):
def spectrum(file_path, audio_index, sr='', f_ratio=0.5, zoom_1='', zoom_2=''):
    global signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)

    ft = scipy.fft.fft(signal)
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sampling_rate, len(magnitude))
    frequency_range = int(len(frequency) * f_ratio)

    plt.figure(figsize=(15, 5))
    plt.plot(frequency[:frequency_range], magnitude[:frequency_range])
    plt.title('Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    if zoom_1 and zoom_2:
        plt.xlim(zoom_1, zoom_2)
        return plt.show()
    else:
        return plt.show()


# 2. SIGNAL PROCESSING FUNCTIONS:
# Function to extract IMFs using Empirical Mode Decomposition (EMD):
def empirical_mode_decomposition(file_path, audio_index, sr='', plot=bool, zoom_1='', zoom_2=''):
    global signal, imfs, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
    emd = EMD()
    imfs = emd(signal)

    # Plotting before and after EMD:
    plots = [signal, imfs[0], imfs[1], imfs[2], imfs[3], imfs[4]]
    plot_title = ['Original Signal', 'IMF_1', 'IMF_2', 'IMF_3', 'IMF_4', 'IMF_5']
    if plot:
        zoom_1 = 1 if zoom_1 == 0 else zoom_1
        plt.figure(figsize=(15, 10))
        if zoom_1 and zoom_2:
            for i in range(6):
                plt.subplot(6, 1, i+1)
                librosa.display.waveshow(plots[i][zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
        else:
            for i in range(6):
                plt.subplot(6, 1, i+1)
                librosa.display.waveshow(plots[i], sr=sampling_rate, alpha=0.6)
        plt.title(plot_title[i])
        plt.tight_layout(pad=2.0)
        return plt.show()
    else:
        return signal, imfs


# Function for Wavelet-Denoising:
def wavelet_denoising(file_path, audio_index, sr='', plot=bool, zoom_1='', zoom_2=''):
    global signal, wl_signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
    wl_signal = denoise_wavelet(signal, method='BayesShrink', mode='soft', wavelet_levels=5,
                                wavelet='sym8', rescale_sigma=True)

    # Plotting the signal before and after Wavelet-Denoising:
    if plot:
        zoom_1 = 1 if zoom_1 == 0 else zoom_1
        plt.figure(figsize=(15, 10))
        if zoom_1 and zoom_2:
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(signal[zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
            plt.title('Before Wavelet Denoising')
            plt.subplot(2, 1, 2)
            librosa.display.waveshow(wl_signal[zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
            plt.title('After Wavelet Denoising')
        else:
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(signal, sr=sampling_rate, alpha=0.6)
            plt.title('Before Wavelet Denoising')
            plt.subplot(2, 1, 2)
            librosa.display.waveshow(wl_signal, sr=sampling_rate, alpha=0.6)
            plt.title('After Wavelet Denoising')

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout(pad=2.0)
        return plt.show()
    else:
        return signal, wl_signal


# Function for Digital Filter - Butterworth:
def digital_filter(file_path, audio_index, order, low_fc='', high_fc='', sr='', plot=bool,
                   zoom_1='', zoom_2='', f_ratio=0.5):
    global signal, filtered_signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)

    if low_fc:
        low = float(low_fc) / (0.5 * float(sampling_rate))
    if high_fc:
        high = float(high_fc) / (0.5 * float(sampling_rate))

    # First condition - Lowpass Filter:
    if low_fc and not high_fc:
        b, a = scipy.signal.butter(order, low, 'lowpass', analog=False)
        filtered_signal = filtfilt(b, a, signal, axis=0)
    # Second Condition - Highpass Filter
    elif high_fc and not low_fc:  # Highpass Filter
        b, a = scipy.signal.butter(order, high, 'highpass', analog=False)
        filtered_signal = filtfilt(b, a, signal, axis=0)
    # Third Condition - Bandpass Filter:
    elif low_fc and high_fc:
        b, a = scipy.signal.butter(order, [high, low], 'bandpass', analog=False)
        filtered_signal = filtfilt(b, a, signal, axis=0)

    # Plotting the Time-Domain and Frequency-Domain (Spectrum) if plot argument is True:
    if plot:
        # Fourier Transform of the original Signal:
        ft_signal = scipy.fft.fft(signal)
        magnitude_signal = np.absolute(ft_signal)
        frequency_signal = np.linspace(0, sampling_rate, len(magnitude_signal))
        frequency_range = int(len(frequency_signal) * f_ratio)

        # Fourier Transform of the filtered Signal:
        ft_filtered_signal = scipy.fft.fft(filtered_signal)
        magnitude_filtered_signal = np.absolute(ft_filtered_signal)
        frequency_filtered_signal = np.linspace(0, sampling_rate, len(magnitude_filtered_signal))
        frequency_range_filtered_signal = int(len(frequency_filtered_signal) * f_ratio)

        # Plotting the original signal and the filtered signal
        plots = [signal, filtered_signal]
        frequencies = [frequency_signal, frequency_filtered_signal]
        magnitudes = [magnitude_signal, magnitude_filtered_signal]
        frequency_ranges = [frequency_range, frequency_range_filtered_signal]
        plot_title = ['Signal before filtering', 'Signal after filtering',
                      'Spectrum before filtering', 'Spectrum after filtering']

        zoom_1 = 1 if zoom_1 == 0 else zoom_1
        if zoom_1 and zoom_2:
            for i in range(2):
                plt.figure(figsize=(15, 10))
                plt.subplot(4, 1, i + 1)
                librosa.display.waveshow(plots[i][zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
                plt.title(plot_title[i])
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
        else:
            for i in range(2):
                plt.figure(figsize=(15, 10))
                plt.subplot(4, 1, i + 1)
                librosa.display.waveshow(plots[i], sr=sampling_rate, alpha=0.6)
                plt.title(plot_title[i])
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
        for i in range(2):
            plt.figure(figsize=(15, 10))
            plt.subplot(4, 1, i + 3)
            plt.plot(frequencies[i][:frequency_ranges[i]], magnitudes[i][:frequency_ranges[i]])
            plt.title(plot_title[i + 2])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')

        plt.tight_layout(pad=2.0)
        return plt.show()
    else:
        return signal, filtered_signal


# Combination of Signal Processing methods:
# Function for EMD + Wavelet-Denoising:
def emd_wavelet(file_path, audio_index, sr='', plot=bool, zoom_1='', zoom_2=''):
    global signal, imfs, emd_wavelet_signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
    # Applying EMD method on the original signal
    emd = EMD()
    imfs = emd(signal)
    # Applying Wavelet-Denoising method to IMF_1 signal
    emd_wavelet_signal = denoise_wavelet(imfs[0], method='BayesShrink', mode='soft', wavelet_levels=5,
                                         wavelet='sym8', rescale_sigma=True)

    # Plotting the original signal, IMF_1 and IMF_1 after wavelet-denoising:
    if plot:
        zoom_1 = 1 if zoom_1 == 0 else zoom_1
        plots = [signal, imfs[0], emd_wavelet_signal]
        plot_title = ['Original Signal', 'IMF_1', 'Signal after applying IMF_1 and Wavelet-Denoising']

        if zoom_1 and zoom_2:
            for i in range(len(plots)):
                plt.figure(figsize=(15, 10))
                plt.subplot(len(plots), 1, i + 1)
                librosa.display.waveshow(plots[i][zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
                plt.plot(plots[i][zoom_1:zoom_2], lw=0.5)
                plt.title(plot_title[i])
        else:
            for i in range(len(plots)):
                plt.figure(figsize=(15, 10))
                plt.subplot(len(plots), 1, i + 1)
                librosa.display.waveshow(plots[i], sr=sampling_rate, alpha=0.6)
                plt.title(plot_title[i])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout(pad=2.0)
        return plt.show()
    else:
        return np.array(signal), np.array(emd_wavelet_signal)


# Function for EMD + Digital-Filter:
def emd_dfilter(file_path, audio_index, order, low_fc='', high_fc='', sr='', plot=bool, zoom_1='',
                zoom_2=''):
    global signal, imfs, emd_dfilter_signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
    # Applying EMD method on the original signal
    emd = EMD()
    imfs = emd(signal)

    # Applying Digital-Filter method to IMF_1 signal
    if low_fc:
        low = float(low_fc) / (0.5 * float(sampling_rate))
    if high_fc:
        high = float(high_fc) / (0.5 * float(sampling_rate))

    # First condition - Lowpass Filter:
    if low_fc and not high_fc:
        b, a = scipy.signal.butter(order, low, 'lowpass', analog=False)
        emd_dfilter_signal = filtfilt(b, a, imfs[0], axis=0)
    # Second Condition - Highpass Filter
    elif high_fc and not low_fc:  # Highpass Filter
        b, a = scipy.signal.butter(order, high, 'highpass', analog=False)
        emd_dfilter_signal = filtfilt(b, a, imfs[0], axis=0)
    # Third Condition - Bandpass Filter:
    elif low_fc and high_fc:
        b, a = scipy.signal.butter(order, [high, low], 'bandpass', analog=False)
        emd_dfilter_signal = filtfilt(b, a, imfs[0], axis=0)

    # Plotting the original signal, imf_1 and imf_1 after digital filter:
    if plot:
        zoom_1 = 1 if zoom_1 == 0 else zoom_1
        plots = [signal, imfs[0], emd_dfilter_signal]
        plot_title = ['Original Signal', 'IMF_1', 'Signal after applying IMF_1 and Digital-Filter']

        if zoom_1 and zoom_2:
            for i in range(len(plots)):
                plt.figure(figsize=(15, 10))
                plt.subplot(3, 1, i + 1)
                librosa.display.waveshow(plots[i][zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
                plt.title(plot_title[i])
        else:
            for i in range(len(plots)):
                plt.figure(figsize=(15, 10))
                plt.subplot(3, 1, i + 1)
                plt.plot(plots[i], lw=0.5)
                plt.title(plot_title[i])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout(pad=2.0)
        return plt.show()
    else:
        return np.array(signal), np.array(emd_dfilter_signal)


# Function for EMD + Wavelet-Denoising + Digital-Filter
def emd_wl_dfilter(file_path, audio_index, order, low_fc='', high_fc='', sr='', plot=bool, zoom_1='',
                   zoom_2=''):
    global signal, imfs, emd_wavelet_signal, emd_wl_dfilter_signal, sampling_rate
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)

    # Applying EMD method on the original signal
    emd = EMD()
    imfs = emd(signal)

    # Applying Wavelet-Denoising method to IMF_1 signal
    emd_wavelet_signal = denoise_wavelet(imfs[0], method='BayesShrink', mode='soft', wavelet_levels=5,
                                         wavelet='sym8', rescale_sigma=True)

    # Applying Digital-Filter method to emd_wavelet_signal
    if low_fc:
        low = float(low_fc) / (0.5 * float(sampling_rate))
    if high_fc:
        high = float(high_fc) / (0.5 * float(sampling_rate))

    # First condition - Lowpass Filter:
    if low_fc and not high_fc:
        b, a = scipy.signal.butter(order, low, 'lowpass', analog=False)
        emd_wl_dfilter_signal = filtfilt(b, a, emd_wavelet_signal, axis=0)
    # Second Condition - Highpass Filter
    elif high_fc and not low_fc:  # Highpass Filter
        b, a = scipy.signal.butter(order, high, 'highpass', analog=False)
        emd_wl_dfilter_signal = filtfilt(b, a, emd_wavelet_signal, axis=0)
    # Third Condition - Bandpass Filter:
    elif low_fc and high_fc:
        b, a = scipy.signal.butter(order, [high, low], 'bandpass', analog=False)
        emd_wl_dfilter_signal = filtfilt(b, a, emd_wavelet_signal, axis=0)

    # plotting the original signal, emd_wavelet signal and emd_wl_dfilter:
    if plot:
        zoom_1 = 1 if zoom_1 == 0 else zoom_1
        plots = [signal, imfs[0], emd_wavelet_signal, emd_wl_dfilter_signal]
        plot_title = ['Original Signal', 'IMF_1', 'Signal after applying IMF_1 and Wavelet-Denoising',
                      'Signal after applying IMF_1, Wavelet-Denoising and Digital-Filter']
        if zoom_1 and zoom_2:
            for i in range(len(plots)):
                plt.figure(figsize=(15, 10))
                plt.subplot(len(plots), 1, i + 1)
                librosa.display.waveshow(plots[i][zoom_1:zoom_2], sr=sampling_rate, alpha=0.6)
                plt.title(plot_title[i])
        else:
            for i in range(len(plots)):
                plt.figure(figsize=(15, 10))
                plt.subplot(len(plots), 1, i + 1)
                librosa.display.waveshow(plots[i], sr=sampling_rate, alpha=0.6)
                plt.title(plot_title[i])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout(pad=2.0)
        return plt.show()
    else:
        return np.array(signal), np.array(emd_wl_dfilter_signal)


# 3. FEATURE EXTRACTION:
# 3.1. Time-Domain Features:
# 3.1.1. Amplitude Envelope:
def amplitude_envelope(processed_signal, frame_size, hop_size, plot=bool):
    ae = np.array([max(processed_signal[i: i + frame_size]) for i in range(0, len(processed_signal), hop_size)])

    # plotting the amplitude envelope:
    if plot:
        # to plot the frames alongside with the processed_signal, the frames required to be converted into time
        frames = range(0, ae.size)
        t = librosa.frames_to_time(frames, hop_length=hop_size)

        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(processed_signal, alpha=0.5)
        plt.title('Amplitude Envelope')
        plt.plot(t, ae, color='r')
        return plt.show()
    else:
        return np.array(ae)


# 3.1.2. Root Mean Square Energy:
def root_mean_square(processed_signal, frame_size, hop_size, plot=bool):
    rms = librosa.feature.rms(processed_signal, frame_length=frame_size, hop_length=hop_size)[0]

    # plotting the rms:
    if plot:
        # to plot the frames alongside with the processed_signal, the frames required to be converted into time
        frames = range(0, rms.size)
        t = librosa.frames_to_time(frames, hop_length=hop_size)

        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(processed_signal, alpha=0.5)
        plt.title('Root Mean Square Energy')
        plt.plot(t, rms, color='r')
        return plt.show()
    else:
        return np.array(rms)


# Zero-Crossing Rate:
def zero_crossing_rate(processed_signal, frame_size, hop_size, plot=bool):
    zcr = librosa.feature.zero_crossing_rate(processed_signal, frame_length=frame_size, hop_length=hop_size)[0]

    if plot:
        frames = range(0, zcr.size)
        t = librosa.frames_to_time(frames, hop_length=hop_size)

        plt.figure(figsize=(15, 5))
        plt.plot(t, zcr, color='r', alpha=0.5)
        plt.title('Zero-Crossing Rate')
        plt.ylim((0, 0.3))
    else:
        return np.array(zcr)


# 3.2. Frequency-Domain Features:
# 3.2.1. Function to extract the Band Energy Ratio:
def band_energy_ratio(processed_signal, frame_size, hop_size, split_frequency, sr, plot=bool):
    # calculating the spectrogram:
    spec = librosa.stft(processed_signal, n_fft=frame_size, hop_length=hop_size)

    # calculating the split frequency bin
    frequency_range = sr / 2
    frequency_delta_per_bin = frequency_range / spec.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)

    # move to the power spectrogram:
    power_spectrogram = np.abs(spec) ** 2
    power_spectrogram = power_spectrogram.T

    # calculating the band energy ratio for each frame:
    ber = []
    for frequencies_in_frame in power_spectrogram:
        sum_power_low_frequencies = np.sum(frequencies_in_frame[:int(split_frequency_bin)])
        sum_power_high_frequencies = np.sum(frequencies_in_frame[int(split_frequency_bin):])
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        ber.append(ber_current_frame)

    # plotting the Band Energy Ratio:
    if plot:
        frames = range(len(ber))
        t = librosa.frames_to_time(frames, hop_length=hop_size)

        plt.figure(figsize=(15, 5))
        plt.plot(t, ber)
        return plt.show()
    else:
        return np.array(ber)


# 3.2.2. Function to extract the Spectral Centroid:
def spectral_centroid(processed_signal, sr, frame_size, hop_size, plot=bool):
    sc = librosa.feature.spectral_centroid(y=processed_signal, sr=sr, n_fft=frame_size, hop_length=hop_size)[0]

    # plotting the spectral centroid:
    if plot:
        frames = range(len(sc))
        t = librosa.frames_to_time(frames)

        plt.figure(figsize=(15, 5))
        plt.plot(t, sc)
        plt.title('Spectral Centroid')
        return plt.show
    else:
        return sc


# 3.2.3. Function to extract the Spectral Bandwidth:
def spectral_bandwidth(processed_signal, sr, frame_size, hop_size, plot=bool):
    sb = librosa.feature.spectral_bandwidth(y=processed_signal, sr=sr, n_fft=frame_size, hop_length=hop_size)[0]

    # plotting the spectral centroid:
    if plot:
        frames = range(len(sb))
        t = librosa.frames_to_time(frames)

        plt.figure(figsize=(15, 5))
        plt.plot(t, sb)
        plt.title('Spectral Bandwidth')
        return plt.show
    else:
        return sb


# 3.3. Time-Frequency representation Features:
# 3.3.1. Function to extract the Spectrogram:
def spectrogram(processed_signal, sr, frame_size, hop_size):  # frame=512, hop=64
    signal_stft = librosa.stft(processed_signal, n_fft=frame_size, hop_length=hop_size)
    signal_stft_log = librosa.power_to_db(np.abs(signal_stft) ** 2)

    # plotting the spectrogram:
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(signal_stft_log, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.f')
    return plt.show()


# 3.3.2. Function to extract the Mel Spectrogram:
def mel_spectrogram(processed_signal, sr, frame_size, hop_size, n_mels):
    signal_mel = librosa.feature.melspectrogram(processed_signal, sr=sr, n_fft=frame_size, hop_length=hop_size,
                                                n_mels=n_mels)
    signal_mel_log = librosa.power_to_db(signal_mel)

    # plotting the Mel Spectrogram:
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(signal_mel_log, sr=sr, hop_length=hop_size, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.f')
    return plt.show()


# 3.3.3. Function to extract Mel Frequency Cepstral Coefficients (MFCCs):
def mel_frequency_cepstral_coefficients(processed_signal, sr, n_mfcc, plot=bool, mfcc_type=''):
    mfccs = librosa.feature.mfcc(processed_signal, n_mfcc=n_mfcc, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Plotting the mfccs, delta_1 and delta_2:
    if plot:
        plt.figure(figsize=(15, 5))
        if mfcc_type == 'mfccs':
            librosa.display.specshow(mfccs, x_axis='time', sr=sampling_rate)
            plt.title('Mel-Frequency Cepstral Coefficients')
        if mfcc_type == 'delta_1':
            librosa.display.specshow(delta_mfccs, x_axis='time', sr=sampling_rate)
            plt.title('Delta_1 MFCCs')
        if mfcc_type == 'delta_2':
            librosa.display.specshow(delta2_mfccs, x_axis='time', sr=sampling_rate)
            plt.title('Delta_2 MFCCs')
        plt.colorbar(format='%+2f')
        return plt.show()
    else:
        return mfccs, delta_mfccs, delta2_mfccs