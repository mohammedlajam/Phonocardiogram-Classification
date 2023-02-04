# Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# signal processing libraries:
import librosa
import librosa.display
import IPython.display as ipd
import scipy.fft
from scipy.signal import welch
from PyEMD import EMD
from skimage.restoration import denoise_wavelet
from scipy.signal import filtfilt
import scipy
import constants as c

import warnings
warnings.filterwarnings('ignore')


# 1. VISUALIZATION FUNCTIONS:
# Function to display the audio file based on audio_path and audio_index:
def display_audio(file_path, audio_index):
    return ipd.Audio(file_path[audio_index])


# Function to extract the signal and sampling frequency:
def signal_extraction(file_path, audio_index, sr=''):
    signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
    print(f'Signal: {signal[:10]}')
    print(f'Signal Shape: {signal.shape}')
    print(f'Sample Rate: {sr}')
    return signal


# Function to visualize the signal in Time-Domain:
def signal_time_domain(file_path, audio_index, sr='', zoom_1='', zoom_2=''):
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


# Visualize a signal in Time-Domain based on Pandas DataFrame:
def visualize_signal(signals, index, sr, zoom_1='', zoom_2=''):
    zoom_1 = 1 if zoom_1 == 0 else zoom_1
    plt.figure(figsize=(15, 5))
    if zoom_1 and zoom_2:
        librosa.display.waveshow(np.array(signals.iloc[index, zoom_1*sr:zoom_2*sr]), sr=sr, alpha=0.6)
        plt.title(f'Zoomed Row Audio Signal ({zoom_1} sec - {zoom_2} sec)')
    else:
        librosa.display.waveshow(np.array(signals.iloc[index, :-1]), sr=sr, alpha=0.6)
        plt.title(f'Complete Row Audio Signal')
    return plt.show()


# Function to display the signal in Frequency-Domain (Spectrum):
def spectrum(file_path, audio_index, sr='', f_ratio=0.5, zoom_1='', zoom_2=''):
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
def empirical_mode_decomposition(file_path, audio_index, sr, plot=False, zoom_1='', zoom_2=''):
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
def wavelet_denoising(file_path, audio_index, sr, plot=False, zoom_1='', zoom_2=''):
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
def digital_filter(file_path, audio_index, sr, order, low_fc='', high_fc='', plot=False,
                   zoom_1='', zoom_2='', f_ratio=0.5):
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
def emd_wavelet(file_path, audio_index, sr, plot=False, zoom_1='', zoom_2=''):
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
def emd_dfilter(file_path, audio_index, sr, order, low_fc='', high_fc='', plot=False, zoom_1='',
                zoom_2=''):
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
def emd_wl_dfilter(file_path, audio_index, sr, order, low_fc='', high_fc='', plot=False, zoom_1='',
                   zoom_2=''):
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


# 3. Slicing:
# Function for slicing the signal:
# The 'signals' Argument is a Pandas DataFrame
# denoise method is either one of the following:
# emd, wavelet_transform, digital_filters, emd_wavelet, emd_dfilters, emd_wl_dfilters
def slice_signals(signals, period, sr, save=False, csv_version=int, denoise_method=''):
    sliced_signals = []
    for i in range(len(signals)):  # iterating over all the rows in the DataFrame
        start = 0
        end = sr * period
        for j in range(8):  # The number of slices in each row
            signal = pd.DataFrame(signals.iloc[i, start:end]).T
            signal['class'] = signals.iloc[i, -1]
            sliced_signals.append(np.array(signal))
            start += sr * period
            end += sr * period

    sliced_signals = [item for elem in sliced_signals for item in elem]

    # converting a list to DataFrame and dropping any row that contains NaN:
    sliced_signals = pd.DataFrame(sliced_signals).dropna()
    sliced_signals.reset_index(drop=True, inplace=True)
    sliced_signals = sliced_signals.rename(columns={sr*period: 'class'})

    # saving the DataFrame into the path of local machine as csv file:
    if save:
        # Checking if the folder exists:
        if os.path.exists(f'{c.REPO_PATH}{c.SIG_PRE_PATH}'):
            if os.path.exists(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}'):
                while os.path.isfile(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}/{denoise_method}_v{csv_version}.csv'):
                    csv_version += 1
                    continue
                else:
                    sliced_signals.to_csv(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}/{denoise_method}_v{csv_version}.csv', index=False)
            else:
                os.mkdir(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}')
                sliced_signals.to_csv(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}/{denoise_method}_v{csv_version}.csv', index=False)
        else:
            os.mkdir(f'{c.REPO_PATH}{c.SIG_PRE_PATH}')
            os.mkdir(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}')
            sliced_signals.to_csv(f'{c.REPO_PATH}{c.SIG_PRE_PATH}/{denoise_method}/{denoise_method}_v{csv_version}.csv', index=False)
    return sliced_signals
