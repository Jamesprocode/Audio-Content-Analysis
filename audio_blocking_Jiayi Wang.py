#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUSI6201 Audio Blocking Assignment

In this assignment, you will convert your pseudo code activity into actual 
code in order to implement an audio "blocking" function. 

Your function may use existing python libraries to load the audio (e.g., scipy, 
librosa)but the rest will be done with just numpy.

Please read the parameters carefully and use the class discussion on Canvas if 
you have questions.

Please note the "pad" condition (i.e., padding with extra zeros so that the 
total number of samples are an integer multiple of the hop size). 
If you did not think of this in class, try working it out in pseudo code first.




 Parameters
 ----------
 audio_input : np.ndarray or str
     Audio input. Should be able to come from either:
     - A NumPy array containing the audio signal.
     - A string path to an audio file (e.g., 'audio.wav').
 sr : int
     Sampling rate. Required if input is np.ndarray
 frame_size : int, optional
     Size of each frame in samples (make default 2048).
 hop_ratio : float, optional    
     Hop size as a ratio of the frame_size (make default .5).
 pad : bool, optional
     If True (default), pads the signal with zeros to ensure all frames are the same length.
     If False, discards the last incomplete frame.

 Returns
 -------
 frames : np.ndarray
     2D array of shape (n_frames, frame_size).
 times : np.ndarray
     Array of start times (in seconds) for each frame.
 """

import numpy as np
import math
import scipy.io.wavfile as wav

def audio_blocking(audio_input, sr=None, frame_size=1024, hop_ratio=0.5, pad=True):
    #analyze sample rate and get audio from scipy

    sr, audio = wav.read(audio_input)

    #down sample to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    #calculate total sample 
    total_samples = len(audio)

    # calculate hop size
    hop_size = int(frame_size * hop_ratio)
    
    if pad:
        #use ceiling to include the tail end of the audio to get total frames
        total_frames = math.ceil((total_samples - frame_size) / hop_size) + 1
        #calculate needed samples and pad amount
        needed_sample = (total_frames - 1) * hop_size + frame_size
        pad_amount = needed_sample - total_samples
        #pad audio
        audio = np.pad(audio, (0, pad_amount), mode='constant')
    else:
        #use floor to discard the tail end of the audio to get total frames
        total_frames = math.floor((total_samples - frame_size) / hop_size) + 1

    print("Total frames:", total_frames)
    print("frame_size:", frame_size)
    frames = np.zeros((total_frames, frame_size))
    times = np.zeros(total_frames)

    for i in range(total_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i, :] = audio[start:end]
        times[i] = start / sr


    return tuple([frames, times])

audio = "/Users/jameswang/workspace/Audio Content Analysis/prelude_cmaj_10s.wav"
print(audio_blocking(audio))

