import numpy as np
import parselmouth
import math
import pandas as pd
import os
# from common_utils import addBack, checkPath
import sys
from scipy.io import wavfile
import warnings
import datetime

warnings.filterwarnings("ignore")

def pitch_contour(src=None, dest=None, tonic=None, k=100, sample_rate=16000, normalize=False, time_step=0.01):
        '''
        Returns a normalized pitch contour at 10 ms intervals
        Parameters
            src (str): loaded audio or source audio file
            dest (str): destination file for contour
            tonic (float): tonic of audio (in Hz)
            k (int): number of divisions per semitone; default to 5
            sample_rate (int): sample rate to load audio in
            normalize (bool): if True, then the pitch contour will be normalised w.r.t. the tonic
            time_step (float): time interval (in s) at which the pitch is going to be extracted
        Returns
            pitchDf (pd.DataFrame): dataframe with time, normalised pitch, energy values
        '''
        if type(src) is np.ndarray:
            # loaded audio
            snd = parselmouth.Sound(src, sample_rate, 0)
        else:
            # audio file name
            snd = parselmouth.Sound(src)

        if tonic:
            min_pitch = tonic*(2**(-5/12)) #lower Pa
            max_pitch = tonic*(2**(19/12))  #higher Pa
            pitch = snd.to_pitch_ac(time_step, min_pitch, 15, True, 0.03, 0.45, 0.01, 0.9, 0.14, max_pitch)
        else:
            pitch = snd.to_pitch_ac(time_step, max_number_of_candidates=15, very_accurate=True\
                , silence_threshold=0.03, voicing_threshold=0.45, octave_cost=0.01, octave_jump_cost=0.9,\
                     voiced_unvoiced_cost=0.14)
        inten = snd.to_intensity(50, time_step, False)

        timestamps = np.arange(0, snd.duration, time_step)
        pitchvals = []
        intensityvals = []
        for t in timestamps:
            pitchvals.append(pitch.get_value_at_time(t) if not math.isnan(pitch.get_value_at_time(t)) else 0)
            intensityvals.append(inten.get_value(t) if not math.isnan(inten.get_value(t)) else 0)

        df = pd.DataFrame(columns=['time', 'pitch', 'energy'])
        for i, f in enumerate(pitchvals):
            df = df.append({'time': timestamps[i],
                            'pitch': f,
                            'energy': intensityvals[i]
                            }, ignore_index=True)
        # silence first and last second, to coverup distortion due to ss
        df.iloc[:int(1/time_step), 1] = 0
        df.iloc[-int(1/time_step):, 1] = 0
        # pdb.set_trace()
        if normalize:
            df = normalize_pitch(df, tonic, k)  # normalize pitch values
        if dest is not None:
            df.to_csv(dest, header=True, index=False)
        return df

def normalize_pitch(df, tonic, k):
        '''
        Replaces the pitch values from dataframe df to normalised pitch values
        Parameters
            df (pd.DataFrame): dataframe with t, p (in Hz) and e values
            tonic (float): tonic in Hz
            k (int): number of divisions per semitone
        
        Returns
            df (pd.DataFrame): dataframe with t, p (normalised) and e values
        '''
        frequency_normalized = [np.round_(1200*math.log2(f/tonic)*(k/100)) if f>0 else -3000 for f in df['pitch']]
        df['pitch'] = frequency_normalized
        return df

if len(sys.argv)==2:
    normalizeFlag=False
    audiofileName=sys.argv[1]
else:
    audiofileName=sys.argv[1]
    tonicfileName=sys.argv[2]
    normalizeFlag=True

sample_rate,array=wavfile.read(audiofileName)

if normalizeFlag:
    pitch_contour_dir='../older_song_pitch_contours'
else:
    pitch_contour_dir='./pitchContours_Hz'
os.makedirs(pitch_contour_dir,exist_ok=True)
output_file_name=os.path.basename(audiofileName).split('.')[0]+'.csv'
output_file_name_full_path=os.path.join(pitch_contour_dir,output_file_name)

print ("Processing ",audiofileName," at ",datetime.datetime.now())

if normalizeFlag:
    with open(tonicfileName,'r') as f:
        line=[line.rstrip() for line in f]
    tonic=float(line[0])
    pitchContour_df=pitch_contour(audiofileName,dest=output_file_name_full_path,tonic=tonic,normalize=normalizeFlag,sample_rate=sample_rate)
else:
    pitchContour_df=pitch_contour(audiofileName,dest=output_file_name_full_path,tonic=0,normalize=normalizeFlag,sample_rate=sample_rate)