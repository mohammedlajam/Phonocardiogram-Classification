# importing libraries:
from glob import glob
import functions
from functions import *

# Loading all the audio files in one list:
folders = ['a', 'b', 'c', 'd', 'e', 'f']

audio_files = []
for folder in folders:
    audio = glob(f'/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/training-{folder}/*.wav')
    audio_files.append(audio)
audio_files = [item for elem in audio_files for item in elem]

# Loading the References:
references = []
for folder in folders:
    signal_csv = pd.read_csv(f'/Users/mohammedlajam/Documents/GitHub/Datasets/Phonocardiogram/PhysioNet_2016/training-{folder}/REFERENCE.csv', header=None)
    signal_csv.columns = ['signal_id', 'class']
    references.append(signal_csv)
signal_classes = pd.concat(references)
signal_classes.reset_index(inplace=True)
signal_classes.drop('index', inplace=True, axis=1)

