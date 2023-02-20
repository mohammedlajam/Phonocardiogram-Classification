"""
Author: Mohammed Lajam

This file contains a collection of helper functions that are commonly used in Phase 2.
These functions provide support for various tasks such as data processing, visualization,and
computation. The purpose of these helper functions is to encapsulate repetitive and complex code
into reusable and modular blocks, making it easier to maintain and improve the overall
functionality of the project.
"""

# Importing libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# signal processing libraries:
import librosa
import librosa.display
import scipy.io.wavfile as wav
from scipy import signal
import IPython.display as ipd
import scipy.fft
from PyEMD import EMD
from skimage.restoration import denoise_wavelet
from scipy.signal import filtfilt
import scipy

import warnings
warnings.filterwarnings('ignore')


# Functions for Signal Preprocessing:
def display_audio(file_path, audio_index):
    """ Function to display the audio file."""
    return ipd.Audio(file_path[audio_index])


def extract_signal(file_path, audio_index, sr, normalization=bool):
    """Function to extract the signal from the audio file and return it in a Numpy array."""
    if normalization:
        audio_signal, sampling_rate = librosa.load(file_path[audio_index], sr=sr)
        return audio_signal, sampling_rate
    else:
        sampling_rate, audio_signal = wav.read(file_path[audio_index])
        # Resample the audio to 1000 Hz
        resampled_signal = signal.resample(audio_signal, int(audio_signal.shape[0] * sr / sampling_rate))
        # Update sampling_rate
        sampling_rate = sr
        return resampled_signal, sampling_rate


# 1. Signal Exploration:
class SignalExploration:
    def __init__(self, audio_signal, sr):
        self.audio_signal = audio_signal
        self.sr = sr
        self.f_ratio = 0.5

    def plot_signal(self, zoom_1=0, zoom_2=None):
        """Function to visualize the Time-Domain of a signal."""
        audio_signal = np.array(self.audio_signal)
        plt.figure(figsize=(15, 5))
        if zoom_2 is not None:
            plt.plot(audio_signal[zoom_1:zoom_2], alpha=0.6)
            plt.xlim(zoom_1, zoom_2)
            plt.title(f'Zoomed Row Audio Signal ({zoom_1} - {zoom_2})')
        else:
            plt.plot(audio_signal[zoom_1:], alpha=0.6)
            plt.title(f'Complete Row Audio Signal')
        plt.show()

    def plot_spectrum(self, zoom_1=0, zoom_2=None):
        """Function to visualize the Spectrum of a signal."""
        audio_signal = np.array(self.audio_signal)
        ft = scipy.fft.fft(audio_signal)
        magnitude = np.abs(ft)
        frequency = np.linspace(0, self.sr, len(magnitude))
        frequency_range = int(len(frequency) * self.f_ratio)

        # plotting the spectrum:
        plt.figure(figsize=(15, 5))
        plt.plot(frequency[:frequency_range], magnitude[:frequency_range], alpha=0.6)
        plt.title('Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

        if zoom_2 is not None:
            plt.xlim(zoom_1, zoom_2)
        plt.show()


# 2. Signal Preprocessing:
class SignalPreprocessing:
    def __init__(self, audio_signal, sr):
        self.audio_signal = audio_signal
        self.sr = sr

    def process_emd(self, n_imf, plot=False, zoom_1=0, zoom_2=None):
        """Function to process Empirical Mode Decomposition."""
        audio_signal = np.array(self.audio_signal)
        emd = EMD()
        imfs = emd(audio_signal, max_imf=n_imf)

        # Plotting before and after EMD:
        if plot:
            audio_signals = [audio_signal, *imfs[:]]
            signal_titles = ['Original Signal', *[f'IMF_{i + 1}' for i in range(5)]]
            plt.figure(figsize=(15, 10))
            for i, (audio_signal, signal_title) in enumerate(zip(audio_signals, signal_titles)):
                plt.subplot(len(audio_signals), 1, i + 1)
                plt.plot(audio_signal[zoom_1:zoom_2], alpha=0.6)
                plt.title(signal_title)
            plt.tight_layout(pad=2.0)
            return plt.show()
        else:
            return np.array(audio_signal), np.array(imfs[-1])

    def process_wavelet_denoising(self, plot=False, zoom_1=0, zoom_2=None):
        """Function to process Wavelet-Denoising."""
        audio_signal = np.array(self.audio_signal)
        wl_signal = denoise_wavelet(audio_signal, method='BayesShrink', mode='soft', wavelet_levels=5,
                                    wavelet='sym8', rescale_sigma=True)

        # Plotting the signal before and after Wavelet-Denoising:
        if plot:
            audio_signals = [audio_signal, wl_signal]
            signal_titles = ['Original Signal', 'Processed Signal']
            plt.figure(figsize=(15, 10))
            for i, (audio_signal, title) in enumerate(zip(audio_signals, signal_titles)):
                plt.subplot(len(audio_signals), 1, i + 1)
                plt.plot(audio_signal[zoom_1:zoom_2], alpha=0.6)
                plt.title(title)
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
            plt.tight_layout(pad=2.0)
            return plt.show()
        else:
            return audio_signal, wl_signal

    def process_digital_filter(self, order, low_fc='', high_fc='', plot=False, zoom_1=0, zoom_2=None, f_ratio=0.5):
        """Function to process Digital-Filters."""
        audio_signal = np.array(self.audio_signal)

        low = float(low_fc) / (0.5 * float(self.sr)) if low_fc else None
        high = float(high_fc) / (0.5 * float(self.sr)) if high_fc else None

        if low_fc and not high_fc:  # Lowpass Filter
            b, a = scipy.signal.butter(order, low, 'lowpass', analog=False)
        elif high_fc and not low_fc:  # Highpass Filter
            b, a = scipy.signal.butter(order, high, 'highpass', analog=False)
        elif low_fc and high_fc:  # Bandpass Filter
            b, a = scipy.signal.butter(order, [high, low], 'bandpass', analog=False)
        else:
            raise ValueError('Either low_fc or high_fc must be provided.')

        filtered_signal = filtfilt(b, a, audio_signal, axis=0)

        # Plotting the Time-Domain and Frequency-Domain (Spectrum) if plot argument is True:
        if plot:
            audio_signals = [audio_signal, filtered_signal]
            magnitudes = []
            frequencies = []
            frequency_ranges = []
            signal_titles = ['Original Signal', 'Filtered Signal',
                             'Spectrum before Filtering', 'Spectrum after Filtering']
            for audio_signal in range(len(audio_signals)):
                ft_signal = scipy.fft.fft(audio_signals[audio_signal])
                magnitude = np.absolute(ft_signal)
                frequency = np.linspace(0, self.sr, len(magnitude))
                frequency_range = int(len(frequency) * f_ratio)
                magnitudes.append(magnitude)
                frequencies.append(frequency)
                frequency_ranges.append(frequency_range)

            plt.figure(figsize=(15, 10))
            for i in range(len(audio_signals)):
                plt.subplot(4, 1, i + 1)
                plt.plot(audio_signals[i][zoom_1:zoom_2], alpha=0.6)
                plt.title(signal_titles[i])
                plt.xlabel('Time')
                plt.ylabel('Amplitude')

            for i in range(len(audio_signals)):
                plt.subplot(4, 1, i + 3)
                plt.plot(frequencies[i][:frequency_ranges[i]], magnitudes[i][:frequency_ranges[i]], alpha=0.6)
                plt.title(signal_titles[i + 2])
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude')

            plt.tight_layout(pad=2.0)
            return plt.show()

        else:
            return np.array(audio_signal), np.array(filtered_signal)

    def process_emd_wl(self, n_imf, plot=False, zoom_1=0, zoom_2=None):
        """Function to process emd and Wavelet-Denoising."""
        # Processing using EMD:
        audio_signal = np.array(self.audio_signal)
        emd = EMD()
        imfs = emd(audio_signal, max_imf=n_imf)

        # Processing using Wavelet-Denoising:
        emd_wavelet_signal = denoise_wavelet(imfs[-1], method='BayesShrink', mode='soft', wavelet_levels=5,
                                             wavelet='sym8', rescale_sigma=True)

        if plot:
            audio_signals = [audio_signal, emd_wavelet_signal]
            signal_titles = ['Original Signal', 'Procesed Signal (EMD & Wavelet-Denoising)']
            plt.figure(figsize=(15, 10))
            for i, (audio_signal, title) in enumerate(zip(audio_signals, signal_titles)):
                plt.subplot(len(audio_signals), 1, i + 1)
                plt.plot(audio_signal[zoom_1:zoom_2], alpha=0.6)
                plt.title(title)
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
            plt.tight_layout(pad=2.0)
            return plt.show()
        else:
            return np.array(audio_signal), np.array(emd_wavelet_signal)

    def process_emd_dfilter(self, n_imf, order, low_fc='', high_fc='', plot=False, zoom_1=0, zoom_2=None, f_ratio=0.5):
        """Function to process emd and Digital-Filters."""
        # Processing using EMD:
        audio_signal = np.array(self.audio_signal)
        emd = EMD()
        imfs = emd(audio_signal, max_imf=n_imf)

        # Processing using Digital Filters:
        low = float(low_fc) / (0.5 * float(self.sr)) if low_fc else None
        high = float(high_fc) / (0.5 * float(self.sr)) if high_fc else None

        if low_fc and not high_fc:  # Lowpass Filter
            b, a = scipy.signal.butter(order, low, 'lowpass', analog=False)
        elif high_fc and not low_fc:  # Highpass Filter
            b, a = scipy.signal.butter(order, high, 'highpass', analog=False)
        elif low_fc and high_fc:  # Bandpass Filter
            b, a = scipy.signal.butter(order, [high, low], 'bandpass', analog=False)
        else:
            raise ValueError('Either low_fc or high_fc must be provided.')

        emd_dfilter_signal = filtfilt(b, a, imfs[-1], axis=0)

        # Plotting the Time-Domain and Frequency-Domain (Spectrum) if plot argument is True:
        if plot:
            audio_signals = [audio_signal, emd_dfilter_signal]
            magnitudes = []
            frequencies = []
            frequency_ranges = []
            signal_titles = ['Signal before EMD & Digital Filter', 'Signal after EMD & Digital Filter',
                             'Spectrum before EMD & Digital Filter', 'Spectrum after EMD & Digital Filter']
            for audio_signal in range(len(audio_signals)):
                ft_signal = scipy.fft.fft(audio_signals[audio_signal])
                magnitude = np.absolute(ft_signal)
                frequency = np.linspace(0, self.sr, len(magnitude))
                frequency_range = int(len(frequency) * f_ratio)
                magnitudes.append(magnitude)
                frequencies.append(frequency)
                frequency_ranges.append(frequency_range)

            plt.figure(figsize=(15, 10))
            for i in range(len(audio_signals)):
                plt.subplot(4, 1, i + 1)
                plt.plot(audio_signals[i][zoom_1:zoom_2], alpha=0.6)
                plt.title(signal_titles[i])
                plt.xlabel('Time')
                plt.ylabel('Amplitude')

            for i in range(len(audio_signals)):
                plt.subplot(4, 1, i + 3)
                plt.plot(frequencies[i][:frequency_ranges[i]], magnitudes[i][:frequency_ranges[i]], alpha=0.6)
                plt.title(signal_titles[i + 2])
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude')

            plt.tight_layout(pad=2.0)
            return plt.show()

        else:
            return np.array(audio_signal), np.array(emd_dfilter_signal)

    def process_emd_wl_dfilter(self, n_imf, order, low_fc='', high_fc='', plot=False, zoom_1=0, zoom_2=None, f_ratio=0.5):
        """Function to process emd, Wavelet-Denoising and Digital-Filters"""
        # Processing using EMD:
        audio_signal = np.array(self.audio_signal)
        emd = EMD()
        imfs = emd(audio_signal, max_imf=n_imf)

        # Processing using Wavelet-Denoising:
        emd_wavelet_signal = denoise_wavelet(imfs[-1], method='BayesShrink', mode='soft', wavelet_levels=5,
                                             wavelet='sym8', rescale_sigma=True)

        # Processing using Digital Filters:
        low = float(low_fc) / (0.5 * float(self.sr)) if low_fc else None
        high = float(high_fc) / (0.5 * float(self.sr)) if high_fc else None

        if low_fc and not high_fc:  # Lowpass Filter
            b, a = scipy.signal.butter(order, low, 'lowpass', analog=False)
        elif high_fc and not low_fc:  # Highpass Filter
            b, a = scipy.signal.butter(order, high, 'highpass', analog=False)
        elif low_fc and high_fc:  # Bandpass Filter
            b, a = scipy.signal.butter(order, [high, low], 'bandpass', analog=False)
        else:
            raise ValueError('Either low_fc or high_fc must be provided.')

        emd_wl_dfilter_signal = filtfilt(b, a, emd_wavelet_signal, axis=0)

        # Plotting the Time-Domain and Frequency-Domain (Spectrum) if plot argument is True:
        if plot:
            audio_signals = [audio_signal, emd_wl_dfilter_signal]
            magnitudes = []
            frequencies = []
            frequency_ranges = []
            signal_titles = ['Signal before precessing', 'Signal after processing',
                             'Spectrum before processing', 'Spectrum after processing']
            for audio_signal in range(len(audio_signals)):
                ft_signal = scipy.fft.fft(audio_signals[audio_signal])
                magnitude = np.absolute(ft_signal)
                frequency = np.linspace(0, self.sr, len(magnitude))
                frequency_range = int(len(frequency) * f_ratio)
                magnitudes.append(magnitude)
                frequencies.append(frequency)
                frequency_ranges.append(frequency_range)

            plt.figure(figsize=(15, 10))
            for i in range(len(audio_signals)):
                plt.subplot(4, 1, i + 1)
                plt.plot(audio_signals[i][zoom_1:zoom_2], alpha=0.6)
                plt.title(signal_titles[i])
                plt.xlabel('Time')
                plt.ylabel('Amplitude')

            for i in range(len(audio_signals)):
                plt.subplot(4, 1, i + 3)
                plt.plot(frequencies[i][:frequency_ranges[i]], magnitudes[i][:frequency_ranges[i]], alpha=0.6)
                plt.title(signal_titles[i + 2])
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude')

            plt.tight_layout(pad=2.0)
            return plt.show()
        else:
            return np.array(audio_signal), np.array(emd_wavelet_signal)


# 3. Slicing:
def slice_signals(audio_signals, sr, period):
    """Function for slicing the signal, The 'signals' Argument is a Pandas DataFrame, which
    contains all the signals.
       denoise method is either one of the following: original_signals, emd, wavelet_transform, digital_filters,
       emd_wavelet, emd_dfilters, emd_wl_dfilters"""
    sliced_signals = []
    for i in range(len(audio_signals)):  # iterating over all the rows in the DataFrame
        start = 0
        end = sr * period
        for j in range(8):  # The number of slices in each row
            audio_signal = pd.DataFrame(audio_signals.iloc[i, start:end]).T
            audio_signal['class'] = audio_signals.iloc[i, -1]
            sliced_signals.append(np.array(audio_signal))
            start += sr * period
            end += sr * period

    sliced_signals = [item for elem in sliced_signals for item in elem]

    # converting a list to DataFrame and dropping any row that contains NaN:
    sliced_signals = pd.DataFrame(sliced_signals).dropna()
    sliced_signals.reset_index(drop=True, inplace=True)
    sliced_signals = sliced_signals.rename(columns={sr * period: 'class'})
    return sliced_signals
