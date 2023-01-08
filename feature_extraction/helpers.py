# Importing libraries:
import numpy as np
import matplotlib.pyplot as plt

# feature extraction libraries:
import librosa
import librosa.display

import warnings
warnings.filterwarnings('ignore')


# 1. Time-Domain Features:
# 1.1. Amplitude Envelope:
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


# 1.2. Root Mean Square Energy:
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


# 1.3. Zero-Crossing Rate:
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


# 2. Frequency-Domain Features:
# 2.1. Function to extract the Band Energy Ratio:
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


# 2.2. Function to extract the Spectral Centroid:
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


# 2.3. Function to extract the Spectral Bandwidth:
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


# 3. Time-Frequency representation Features:
# 3.1. Function to extract the Spectrogram:
def spectrogram(processed_signal, sr, frame_size, hop_size):  # frame=512, hop=64
    signal_stft = librosa.stft(processed_signal, n_fft=frame_size, hop_length=hop_size)
    signal_stft_log = librosa.power_to_db(np.abs(signal_stft) ** 2)

    # plotting the spectrogram:
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(signal_stft_log, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.f')
    return plt.show()


# 3.2. Function to extract the Mel Spectrogram:
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


# 3.3. Function to extract Mel Frequency Cepstral Coefficients (MFCCs):
def mel_frequency_cepstral_coefficients(processed_signal, sr, n_mfcc, plot=bool, mfcc_type=''):
    mfccs = librosa.feature.mfcc(processed_signal, n_mfcc=n_mfcc, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

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
        return mfccs, delta_mfccs, delta2_mfccs
