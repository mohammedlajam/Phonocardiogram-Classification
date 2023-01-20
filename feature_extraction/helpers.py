# Importing libraries:
import numpy as np
import matplotlib.pyplot as plt

# feature extraction libraries:
import librosa
import librosa.display
import scipy.signal

import warnings
warnings.filterwarnings('ignore')


# 1. Time-Domain Features:
# 1.1. Amplitude Envelope:
def amplitude_envelope(signal, frame_size, hop_size, plot=bool, des_stats=bool):
    ae = np.array([max(signal[i: i + frame_size]) for i in range(0, len(signal), hop_size)])

    # plotting the amplitude envelope:
    if plot:
        # to plot the frames alongside with the processed_signal, the frames required to be converted into time
        frames = range(0, ae.size)
        t = librosa.frames_to_time(frames, hop_length=hop_size)

        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(signal, alpha=0.5)
        plt.title('Amplitude Envelope')
        plt.plot(t, ae, color='r')
        return plt.show()
    # returning the descriptive statistics of the amplitude envelope:
    elif des_stats:
        ae = np.array(ae)
        maximum = np.max(ae)
        minimum = np.min(ae)
        median = np.median(ae)
        mean = np.mean(ae)
        std = np.std(ae)
        return maximum, minimum, mean, median, std
    # returning the amplitude envelope:
    else:
        return np.array(ae)


# 1.2. Root Mean Square Energy:
def root_mean_square(signal, frame_size, hop_size, plot=bool, des_stats=bool):
    rms = librosa.feature.rms(signal, frame_length=frame_size, hop_length=hop_size)[0]

    # plotting the rms:
    if plot:
        # to plot the frames alongside with the processed_signal, the frames required to be converted into time
        frames = range(0, rms.size)
        t = librosa.frames_to_time(frames, hop_length=hop_size)

        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(signal, alpha=0.5)
        plt.title('Root Mean Square Energy')
        plt.plot(t, rms, color='r')
        return plt.show()
    elif des_stats:
        rms = np.array(rms)
        maximum = np.max(rms)
        minimum = np.min(rms)
        median = np.median(rms)
        mean = np.mean(rms)
        std = np.std(rms)
        return maximum, minimum, mean, median, std

    else:
        return np.array(rms)


# 1.3. Zero-Crossing Rate:
def zero_crossing_rate(signal, frames=bool, frame_size='', hop_size='', plot=bool, des_stats=bool):
    # retrieve the Zero-Crossing Rate for each frame by using frame_size and hop_size:
    signal = np.array(signal)
    if frames:
        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=frame_size, hop_length=hop_size)[0]

        # plotting the Zero-Crossing Rate over the frames:
        if plot:
            frames_range = range(0, zcr.size)
            t = librosa.frames_to_time(frames_range, hop_length=hop_size)

            plt.figure(figsize=(15, 5))
            plt.plot(t, zcr, color='r', alpha=0.5)
            plt.title('Zero-Crossing Rate')
            plt.ylim((0, 0.3))
            return plt.show()

        elif des_stats:
            zcr = np.array(zcr)
            maximum = np.max(zcr)
            minimum = np.min(zcr)
            median = np.median(zcr)
            mean = np.mean(zcr)
            std = np.std(zcr)
            return maximum, minimum, mean, median, std
        else:
            return np.array(zcr)
    # retrieve the Zero-Crossing Rate for the complete signal:
    else:
        zcr = librosa.zero_crossings(signal)
        zcr = np.sum(zcr) / len(zcr)
        return zcr


# 1.4. Descriptive Statistics:
def descriptive_statistics(signal):
    maximum = np.max(signal)
    minimum = np.min(signal)
    median = np.median(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    return maximum, minimum, mean, median, std


# 1.5. Function to calculate the Energy  and the total power of the signal:
def energy_power(signal):
    time_interval = len(signal)
    energy = np.sum((abs(signal))**2)
    total_power = energy / time_interval
    return energy, total_power


# 2. Frequency-Domain Features:
# 2.1. Function to extract the Band Energy Ratio:
def band_energy_ratio(signal, frame_size, hop_size, split_frequency, sr, plot=bool, des_stats=bool):
    # calculating the spectrogram:
    signal = np.array(signal)
    spec = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size)

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
    elif des_stats:
        ber = np.array(ber)
        maximum = np.max(ber)
        minimum = np.min(ber)
        median = np.median(ber)
        mean = np.mean(ber)
        std = np.std(ber)
        return maximum, minimum, mean, median, std
    else:
        return np.array(ber)


# 2.2. Function to extract the Spectral Centroid:
def spectral_centroid(signal, sr, frame_size, hop_size, plot=bool, des_stats=bool):
    signal = np.array(signal)
    sc = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=frame_size, hop_length=hop_size)[0]

    # plotting the spectral centroid:
    if plot:
        frames = range(len(sc))
        t = librosa.frames_to_time(frames)

        plt.figure(figsize=(15, 5))
        plt.plot(t, sc)
        plt.title('Spectral Centroid')
        return plt.show
    elif des_stats:
        sc = np.array(sc)
        maximum = np.max(sc)
        minimum = np.min(sc)
        median = np.median(sc)
        mean = np.mean(sc)
        std = np.std(sc)
        return maximum, minimum, mean, median, std
    else:
        return sc


# 2.3. Function to extract the Spectral Bandwidth:
def spectral_bandwidth(signal, sr, frame_size, hop_size, plot=bool, des_stats=bool):
    signal = np.array(signal)
    sb = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=frame_size, hop_length=hop_size)[0]

    # plotting the spectral centroid:
    if plot:
        frames = range(len(sb))
        t = librosa.frames_to_time(frames)

        plt.figure(figsize=(15, 5))
        plt.plot(t, sb)
        plt.title('Spectral Bandwidth')
        return plt.show
    elif des_stats:
        sb = np.array(sb)
        maximum = np.max(sb)
        minimum = np.min(sb)
        median = np.median(sb)
        mean = np.mean(sb)
        std = np.std(sb)
        return maximum, minimum, mean, median, std
    else:
        return sb


# 2.3. Function to extract the amplitude and the frequency of the peak value:
def peak_frequency(signal, sr, plot=False):
    ft = scipy.fft.fft(signal)
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sr, len(magnitude))
    frequency_range = int(len(frequency) * 0.5)

    peak_amplitude = np.max(magnitude)  # Maximum amplitude
    peak_freq = frequency[np.argmax(magnitude)]  # The frequency of the maximum amplitude

    # plotting the spectrum:
    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(frequency[:frequency_range], magnitude[:frequency_range])
        plt.title('Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()
        return peak_amplitude, peak_freq
    else:
        return peak_amplitude, peak_freq


# 3. Time-Frequency representation Features:
# 3.1. Function to extract the Spectrogram:
def spectrogram(signal, sr, frame_size, hop_size):  # frame=512, hop=64
    signal_stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size)
    signal_stft_log = librosa.power_to_db(np.abs(signal_stft) ** 2)

    # plotting the spectrogram:
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(signal_stft_log, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.f')
    return plt.show()


# 3.2. Function to extract the Mel Spectrogram:
def mel_spectrogram(signal, sr, frame_size, hop_size, n_mels):
    signal_mel = librosa.feature.melspectrogram(signal, sr=sr, n_fft=frame_size, hop_length=hop_size,
                                                n_mels=n_mels)
    signal_mel_log = librosa.power_to_db(signal_mel)

    # plotting the Mel Spectrogram:
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(signal_mel_log, sr=sr, hop_length=hop_size, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.f')
    return plt.show()


# 3.3. Function to extract Mel Frequency Cepstral Coefficients (MFCCs):
def mel_frequency_cepstral_coefficients(signal, sr, n_mfcc, plot=bool, mfcc_type=''):
    mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    maximum = np.max(mfccs)
    minimum = np.min(mfccs)
    mean = np.mean(mfccs)
    median = np.median(mfccs)
    std = np.std(mfccs)

    # Plotting the mfccs, delta_1 and delta_2:
    if plot:
        plt.figure(figsize=(15, 5))
        if mfcc_type == 'mfccs':
            librosa.display.specshow(mfccs, x_axis='time', sr=sr)
            plt.title('Mel-Frequency Cepstral Coefficients')
        if mfcc_type == 'delta_1':
            librosa.display.specshow(delta_mfccs, x_axis='time', sr=sr)
            plt.title('Delta_1 MFCCs')
        if mfcc_type == 'delta_2':
            librosa.display.specshow(delta2_mfccs, x_axis='time', sr=sr)
            plt.title('Delta_2 MFCCs')
        plt.colorbar(format='%+2f')
        return plt.show()
    else:
        return maximum, minimum, mean, median, std