# Importing libraries:
import numpy as np
import matplotlib.pyplot as plt
import constants as c

# feature extraction libraries:
import librosa
import librosa.display
import scipy.signal
import pywt

import warnings
warnings.filterwarnings('ignore')


# 1. Time-Domain Features:
class TimeDomainFeatures:
    def __init__(self, audio_signal, frame_size, hop_size):
        self.audio_signal = audio_signal
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract_descriptive_statistics(self):
        """Function to extract the Descriptive Statistics of a signal"""
        maximum = np.max(self.audio_signal)
        minimum = np.min(self.audio_signal)
        median = np.median(self.audio_signal)
        mean = np.mean(self.audio_signal)
        std = np.std(self.audio_signal)
        return maximum, minimum, mean, median, std

    def extract_energy_power(self):
        """Function to calculate the Energy  and the total power of the signal."""
        time_interval = len(self.audio_signal)
        energy = np.sum((abs(self.audio_signal)) ** 2)
        total_power = energy / time_interval
        return energy, total_power

    def extract_amplitude_envelope(self, plot=False, des_stats=bool):
        """Function to extract the Amplitude Envelope of each frame of a signal. It returns
        either a plot, descriptive statistics or an array of amplitude envelope"""
        audio_signal = np.array(self.audio_signal)
        ae = np.array([max(audio_signal[i: i + self.frame_size]) for i in range(0, len(audio_signal), self.hop_size)])

        # Plotting the amplitude envelope:
        if plot:
            # to plot the frames alongside with the processed_signal, the frames required to be converted into time
            frames = range(0, ae.size)
            t = librosa.frames_to_time(frames, hop_length=self.hop_size)

            plt.figure(figsize=(15, 5))
            librosa.display.waveshow(audio_signal, alpha=0.5)
            plt.title('Amplitude Envelope')
            plt.plot(t, ae, color='r')
            return plt.show()
        # Returning the descriptive statistics of the amplitude envelope:
        elif des_stats:
            ae = np.array(ae)
            maximum = np.max(ae)
            minimum = np.min(ae)
            median = np.median(ae)
            mean = np.mean(ae)
            std = np.std(ae)
            return maximum, minimum, mean, median, std
        # Returning the amplitude envelope:
        else:
            return np.array(ae)

    def extract_root_mean_square(self, plot=False, des_stats=bool):
        """Function to extract the Root Mean Square Energy of each frame of a signal. It returns
        either a plot, descriptive statistics or an array of Root Mean Square Energy"""
        audio_signal = np.array(self.audio_signal)
        rms = librosa.feature.rms(audio_signal, frame_length=self.frame_size, hop_length=self.hop_size)[0]

        # Plotting the rms:
        if plot:
            # to plot the frames alongside with the processed_signal, the frames required to be converted into time
            frames = range(0, rms.size)
            t = librosa.frames_to_time(frames, hop_length=self.hop_size)

            plt.figure(figsize=(15, 5))
            librosa.display.waveshow(audio_signal, alpha=0.5)
            plt.title('Root Mean Square Energy')
            plt.plot(t, rms, color='r')
            return plt.show()
        # Returning the descriptive statistics of the Root Mean Square Energy:
        elif des_stats:
            rms = np.array(rms)
            maximum = np.max(rms)
            minimum = np.min(rms)
            median = np.median(rms)
            mean = np.mean(rms)
            std = np.std(rms)
            return maximum, minimum, mean, median, std
        # Returning the Root Mean Square Energy:
        else:
            return np.array(rms)

    def extract_zero_crossing_rate(self, frames=bool, plot=False, des_stats=bool):
        """Function to extract the Zero-Crossing Rate of each frame of a signal. It returns
        either a plot, descriptive statistics or an array of Zero-Crossing Rate or Zero-Crossing
        Rate of the complete signal. the 'frames' indicates whether the calculations by frames
        or for the complete signal."""
        # retrieve the Zero-Crossing Rate for each frame by using frame_size and hop_size:
        audio_signal = np.array(self.audio_signal)
        if frames:
            zcr = librosa.feature.zero_crossing_rate(audio_signal,
                                                     frame_length=self.frame_size,
                                                     hop_length=self.hop_size)[0]

            # Plotting the Zero-Crossing Rate over the frames:
            if plot:
                frames_range = range(0, zcr.size)
                t = librosa.frames_to_time(frames_range, hop_length=self.hop_size)

                plt.figure(figsize=(15, 5))
                plt.plot(t, zcr, color='r', alpha=0.5)
                plt.title('Zero-Crossing Rate')
                plt.ylim((0, 0.3))
                return plt.show()
            # Returning the descriptive statistics of the Zero-Crossing Rate:
            elif des_stats:
                zcr = np.array(zcr)
                maximum = np.max(zcr)
                minimum = np.min(zcr)
                median = np.median(zcr)
                mean = np.mean(zcr)
                std = np.std(zcr)
                return maximum, minimum, mean, median, std
            # Returning the Zero-Crossing Rate:
            else:
                return np.array(zcr)
        # Retrieve the Zero-Crossing Rate for the complete signal:
        else:
            zcr = librosa.zero_crossings(audio_signal)
            zcr = np.sum(zcr) / len(zcr)
            return zcr


# 2. Frequency-Domain Features:
class FrequencyDomainFeatures:
    def __init__(self, audio_signal, sr, frame_size, hop_size):
        self.audio_signal = audio_signal
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract_spectrum_features(self, plot=False):
        """Function to extract the spectrum features of a signal. It returns the amplitude and
        the frequency values of the peak value."""
        audio_signal = np.array(self.audio_signal)
        ft = scipy.fft.fft(audio_signal)
        magnitude = np.absolute(ft)
        frequency = np.linspace(0, self.sr, len(magnitude))
        frequency_range = int(len(frequency) * 0.5)

        peak_amplitude = np.max(magnitude)  # Maximum amplitude
        peak_frequency = frequency[np.argmax(magnitude)]  # The frequency of the maximum amplitude

        # Plotting the spectrum:
        if plot:
            plt.figure(figsize=(15, 5))
            plt.plot(frequency[:frequency_range], magnitude[:frequency_range])
            plt.title('Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            return plt.show()
        else:
            return peak_amplitude, peak_frequency

    def extract_band_energy_ratio(self, split_frequency, plot=False, des_stats=bool):
        """Function to extract the Band Energy Ratio of each frame of a signal. It returns
        either a plot, descriptive statistics or an array of Band Energy Ratio."""
        # calculating the spectrogram:
        audio_signal = np.array(self.audio_signal)
        spec = librosa.stft(audio_signal, n_fft=self.frame_size, hop_length=self.hop_size)

        # calculating the split frequency bin
        frequency_range = self.sr / 2
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

        # Plotting the Band Energy Ratio:
        if plot:
            frames = range(len(ber))
            t = librosa.frames_to_time(frames, hop_length=self.hop_size)

            plt.figure(figsize=(15, 5))
            plt.plot(t, ber)
            return plt.show()
        # Returning the descriptive statistics of the Band Energy Ratio:
        elif des_stats:
            ber = np.array(ber)
            maximum = np.max(ber)
            minimum = np.min(ber)
            median = np.median(ber)
            mean = np.mean(ber)
            std = np.std(ber)
            return maximum, minimum, mean, median, std
        # Returning the Band Energy Ratio:
        else:
            return np.array(ber)

    def extract_spectral_centroid(self, plot=False, des_stats=bool):
        """Function to extract the Spectral Centroid of each frame of a signal. It returns
        either a plot, descriptive statistics or an array of Spectral Centroid."""
        audio_signal = np.array(self.audio_signal)
        sc = librosa.feature.spectral_centroid(y=audio_signal,
                                               sr=self.sr,
                                               n_fft=self.frame_size,
                                               hop_length=self.hop_size)[0]

        # Plotting the Spectral Centroid:
        if plot:
            frames = range(len(sc))
            t = librosa.frames_to_time(frames)

            plt.figure(figsize=(15, 5))
            plt.plot(t, sc)
            plt.title('Spectral Centroid')
            return plt.show
        # Returning the descriptive statistics of the Spectral Centroid:
        elif des_stats:
            sc = np.array(sc)
            maximum = np.max(sc)
            minimum = np.min(sc)
            median = np.median(sc)
            mean = np.mean(sc)
            std = np.std(sc)
            return maximum, minimum, mean, median, std
        # Returning the Spectral Centroid:
        else:
            return sc

    def extract_spectral_bandwidth(self, plot=False, des_stats=bool):
        """Function to extract the Spectral Centroid of each frame of a signal. It returns
        either a plot, descriptive statistics or an array of Spectral Bandwidth."""
        audio_signal = np.array(self.audio_signal)
        sb = librosa.feature.spectral_bandwidth(y=audio_signal,
                                                sr=self.sr,
                                                n_fft=self.frame_size,
                                                hop_length=self.hop_size)[0]

        # Plotting the Spectral Bandwidth:
        if plot:
            frames = range(len(sb))
            t = librosa.frames_to_time(frames)

            plt.figure(figsize=(15, 5))
            plt.plot(t, sb)
            plt.title('Spectral Bandwidth')
            return plt.show
        # Returning the descriptive statistics of the Spectral Bandwidth:
        elif des_stats:
            sb = np.array(sb)
            maximum = np.max(sb)
            minimum = np.min(sb)
            median = np.median(sb)
            mean = np.mean(sb)
            std = np.std(sb)
            return maximum, minimum, mean, median, std
        # Returning the Spectral Bandwidth:
        else:
            return sb


# 3. Time-Frequency representation Features:
class TimeFrequencyDomainFeatures:
    def __init__(self, audio_signal, sr, frame_size, hop_size):
        self.audio_signal = audio_signal
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract_spectrogram(self, plot=False, save=False, img_ref=None):
        """Funtion to extract the Spectrogram of a signal. It either returns a plot or save
        the image in 'data/extracted_features/images' directory."""
        audio_signal = np.array(self.audio_signal)
        signal_stft = librosa.stft(audio_signal, n_fft=self.frame_size, hop_length=self.hop_size)
        signal_stft_log = librosa.power_to_db(np.abs(signal_stft) ** 2)

        # plotting the spectrogram:
        if plot:
            plt.figure(figsize=(15, 5))
            librosa.display.specshow(signal_stft_log, sr=self.sr, hop_length=self.hop_size, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.f')
            plt.title('Spectrogram')
            return plt.show()
        # Saving the plot in the defined path
        elif save:
            if img_ref is None:
                raise ValueError("'img_ref' argument must be specified")
            figure = plt.figure(figsize=(15, 5))
            ax = plt.axes()
            ax.set_axis_off()
            librosa.display.specshow(signal_stft_log, sr=self.sr, hop_length=self.hop_size, x_axis='time', y_axis='log')
            plt.savefig(f'{c.FEATURE_EXTRACTION_PATH}/images/spectrogram/{img_ref}.png')
            return plt.close(figure)

    def extract_mel_spectrogram(self, n_mels: int, plot=False, save=False, img_ref=None):
        """Funtion to extract the Mel-Spectrogram of a signal. It either returns a plot or
        save the image in 'data/extracted_features/images' directory."""
        audio_signal = np.array(self.audio_signal)
        signal_mel = librosa.feature.melspectrogram(audio_signal,
                                                    sr=self.sr,
                                                    n_fft=self.frame_size,
                                                    hop_length=self.hop_size,
                                                    n_mels=n_mels)
        signal_mel_log = librosa.power_to_db(signal_mel)

        # Plotting the Mel Spectrogram:
        if plot:
            plt.figure(figsize=(15, 5))
            librosa.display.specshow(signal_mel_log, sr=self.sr, hop_length=self.hop_size, x_axis='time', y_axis='mel')
            plt.title('Mel Spectrogram')
            plt.colorbar(format='%+2.f')
            return plt.show()
        # Saving the plot in the defined path:
        elif save:
            if img_ref is None:
                raise ValueError("'img_ref' argument must be specified")
            figure = plt.figure(figsize=(15, 5))
            ax = plt.axes()
            ax.set_axis_off()
            librosa.display.specshow(signal_mel_log, sr=self.sr, hop_length=self.hop_size, x_axis='time', y_axis='mel')
            plt.savefig(f'{c.FEATURE_EXTRACTION_PATH}/images/mel_spectrogram/{img_ref}.png')
            return plt.close(figure)

    def extract_mfccs(self, n_mfcc: int, mfcc_type: str, plot=False, save=False, img_ref=None, des_stats=False):
        """Funtion to extract the MFCCs of a signal. 'mfcc_type' is either 'mfccs', 'delta_1'
        or 'delta_2'. It either returns a plot, save the image in 'data/extracted_features/images'
        directory or the descriptive statistics."""
        audio_signal = np.array(self.audio_signal)
        mfccs = librosa.feature.mfcc(audio_signal, n_mfcc=n_mfcc, sr=self.sr)

        if mfcc_type == 'mfccs':
            if plot:
                plt.figure(figsize=(15, 5))
                librosa.display.specshow(mfccs, x_axis='time', sr=self.sr)
                plt.title('Mel-Frequency Cepstral Coefficients')
                return plt.show()
            elif save:
                if img_ref is None:
                    raise ValueError("'img_ref' argument must be specified")
                figure = plt.figure(figsize=(15, 5))
                ax = plt.axes()
                ax.set_axis_off()
                librosa.display.specshow(mfccs, x_axis='time', sr=self.sr)
                plt.savefig(f'{c.FEATURE_EXTRACTION_PATH}/images/mfccs/mfcc/{img_ref}.png')
                return plt.close(figure)
            elif des_stats:
                return np.max(mfccs), np.min(mfccs), np.mean(mfccs), np.median(mfccs), np.std(mfccs)
            else:
                return np.array(mfccs)
        elif mfcc_type == 'delta_1':
            delta_1 = librosa.feature.delta(mfccs)
            if plot:
                plt.figure(figsize=(15, 5))
                librosa.display.specshow(delta_1, x_axis='time', sr=self.sr)
                plt.title('MFCCs - Delta_1')
                return plt.show()
            elif save:
                if img_ref is None:
                    raise ValueError("'img_ref' argument must be specified")
                figure = plt.figure(figsize=(15, 5))
                ax = plt.axes()
                ax.set_axis_off()
                librosa.display.specshow(delta_1, x_axis='time', sr=self.sr)
                plt.savefig(f'{c.FEATURE_EXTRACTION_PATH}/images/mfccs/delta_1/{img_ref}.png')
                return plt.close(figure)
            elif des_stats:
                return np.max(delta_1), np.min(delta_1), np.mean(delta_1), np.median(delta_1), np.std(delta_1)
            else:
                return np.array(delta_1)
        elif mfcc_type == 'delta_2':
            delta_2 = librosa.feature.delta(mfccs, order=2)
            if plot:
                plt.figure(figsize=(15, 5))
                librosa.display.specshow(delta_2, x_axis='time', sr=self.sr)
                plt.title('MFCCs - Delta_2')
                return plt.show()
            elif save:
                if img_ref is None:
                    raise ValueError("'img_ref' argument must be specified")
                figure = plt.figure(figsize=(15, 5))
                ax = plt.axes()
                ax.set_axis_off()
                librosa.display.specshow(delta_2, x_axis='time', sr=self.sr)
                plt.savefig(f'{c.FEATURE_EXTRACTION_PATH}/images/mfccs/delta_2/{img_ref}.png')
                return plt.close(figure)
            elif des_stats:
                return np.max(delta_2), np.min(delta_2), np.mean(delta_2), np.median(delta_2), np.std(delta_2)
            else:
                return np.array(delta_2)

    def extract_cwt_scalogram(self, num_scales: int, wavelet_family: str, plot=False, save=False, img_ref=None):
        """Funtion to extract the cwt_scalogram of a signal. 'wavelet_family': 'gaus1', 'cgau1',
        'cmor', 'fbsp', 'mexh', 'morl' or 'shan'. It either returns a plot or save the image
        in 'data/extracted_features/images' directory."""
        scales_range = np.arange(1, num_scales)
        coefficients, frequencies = pywt.cwt(data=self.audio_signal, scales=scales_range, wavelet=wavelet_family)

        # plotting the Scalogram:
        if plot:
            plt.figure(figsize=(15, 5))
            plt.imshow(abs(coefficients), extent=[0, 200, 30, 1], interpolation='bilinear',
                       cmap='bone', aspect='auto', vmax=abs(coefficients).max(),
                       vmin=-abs(coefficients).max())
            plt.gca().invert_yaxis()
            plt.yticks(np.arange(1, 31, 1))
            plt.xticks(np.arange(0, 201, 10))
            return plt.show()

        # Saving the Scalogram in the defined path:
        elif save:
            if img_ref is None:
                raise ValueError("'img_ref' argument must be specified.")
            figure = plt.figure(figsize=(15, 5))
            ax = plt.axes()
            ax.set_axis_off()
            plt.imshow(abs(coefficients), extent=[0, 200, 30, 1], interpolation='bilinear',
                       cmap='bone', aspect='auto', vmax=abs(coefficients).max(),
                       vmin=-abs(coefficients).max())
            plt.gca().invert_yaxis()
            plt.savefig(f'{c.FEATURE_EXTRACTION_PATH}/images/scalogram/{img_ref}.png')
            plt.close(figure)

        else:
            return np.array(coefficients), np.array(frequencies)

    def extract_dwt_coefficients(self, dwt_levels=False, plot=False, des_stats=False):
        """Funtion to extract the dwt_coefficients of a signal. if 'dwt_levels' is True, it
        extract five levels of the dwt, otherwise only one level. It either returns a plot or
        save the image in 'data/extracted_features/images' directory."""
        audio_signal = self.audio_signal
        if dwt_levels:
            coefficients = pywt.wavedec(data=audio_signal, wavelet='bior3.1', level=5, mode='periodic')
            ca5, cd5, cd4, cd3, cd2, cd1 = coefficients
            # Plotting the coefficients of the 5 levels:
            if plot:
                audio_signals = [audio_signal, cd1, cd2, cd3, cd4, cd5]
                plot_titles = ['Original Signal', 'cd1', 'cd2', 'cd3', 'cd4', 'cd5']
                fig, axs = plt.subplots(len(audio_signals), 1, figsize=(15, 10))
                for i, (signal, title) in enumerate(zip(audio_signals, plot_titles)):
                    axs[i].plot(signal)
                    axs[i].set_title(title)
                    axs[i].set_xlabel('Time')
                    axs[i].set_ylabel('Amplitude')
                    fig.tight_layout(pad=2.0)
                return plt.show()
            else:
                return ca5, cd5, cd4, cd3, cd2, cd1
        else:
            ca, cd = pywt.dwt(data=audio_signal, wavelet='coif1')
            if des_stats:
                ca_max = np.max(ca)
                ca_min = np.min(ca)
                ca_mean = np.mean(ca)
                ca_median = np.median(ca)
                ca_std = np.std(cd)
                cd_max = np.max(cd)
                cd_min = np.min(cd)
                cd_mean = np.mean(cd)
                cd_median = np.median(cd)
                cd_std = np.std(cd)
                return ca_max, ca_min, ca_mean, ca_median, ca_std, cd_max, cd_min, cd_mean, cd_median, cd_std
            else:
                return ca, cd
