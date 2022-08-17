#!/usr/bin/env python
# author: rajeshr
# github: its-rajesh
# coding: utf-8

#----------------------------------------------------------------------------------
# # Music Speech Classifier
#----------------------------------------------------------------------------------


### INPUTS:
### folder path: folder contains the test samples
### RETURNS:
### Creates a folder 'Results' in the same path. (Only Speech Extracted)
#----------------------------------------------------------------------------------


# INPUTS:
# download the pretrained msclassifier model from () and give the path here
model_path = '/home/rajesh/Desktop/Projects/Music-Speech-Separation/msclassifier.h5'
# copy the test sample folder path and paste here
folder_path = '/home/rajesh/Desktop/Projects/Music-Speech-Separation/Test Data/overlay'

#----------------------------------------------------------------------------------


from keras.models import load_model
import librosa as lb
import numpy as np
import os
from pydub import AudioSegment


# ### LOADING PRETRAINED MODEL

msclassifier = load_model(model_path)


'''
THIS FUNCTION CREATES CHUNCKS OF THE GIVEN AUDIO FILE
INPUTS: AUDIO (Can change the chunks length in seconds as indicated)
RETURNS: CHUNCKS
'''
def createchunks(audio):
  start = 0
  stop = 4000 # for 0.5 seconds (8Khz): 1 sec has 8000 samples.
  chunks = []
  for i in range(len(audio)//4000):
    chunks.append(audio[start:stop])
    start = stop
    stop = start+4000

  return chunks


# ### UPLOAD FOLDER

files = os.listdir(folder_path)
print('{} files found in the folder ({},{},...)'.format(len(files), files[0], files[1]))



k = 1
for audio in files:
    
    path = folder_path+'/'+audio
    sample_audio, sr  = lb.load(path, sr=8000, mono=True)
    
    chunks_sample = np.array(createchunks(sample_audio))
    spect_sample = []
    for i in chunks_sample:
        spect_sample.append(np.abs(lb.stft(i,n_fft=512))) #window length = nfft and hop length = win length //4

    spect_sample = np.array(spect_sample)
    
    res = msclassifier.predict(spect_sample)
    res = np.argmax(res, axis=1)
    
    seconds = 0
    time_split = []
    for i in range(len(res)):
        seconds += 0.5
        if res[i] == 1:
            time_split.append(seconds)

    time_index = []
    time_index.append(time_split[0])
    for i in range(len(time_split)-1):
        if time_split[i+1] != time_split[i]+0.5:
            time_index.append(time_split[i])
            time_index.append(time_split[i+1])

    time_index.append(time_split[-1])
    
    time_tuple = []
    for i in range(0, len(time_index)-1, 2):
        if time_index[i] != 0 and time_index[i] != 0.5:
            time_tuple.append(((time_index[i])*1000, (time_index[i+1])*1000)) #converting into msec
        else:
            time_tuple.append(((time_index[i])*1000, (time_index[i+1])*1000))

    
    sample_audio = AudioSegment.from_file(path, format="wav")

    trimmed_audio = []
    for sec in time_tuple:
        trimmed_audio.append(sample_audio[int(sec[0]):int(sec[1])])

    extracted = trimmed_audio[0]
    for i in range(1, len(trimmed_audio)):
        extracted = extracted+trimmed_audio[i]
    
    
    result_path = folder_path+'/Results'
    try:
        os.mkdir(result_path)
    except:
        pass
    extracted.export(result_path+'/extracted'+str(k)+'.wav', format="wav")
    k += 1
    
print("Results can be seen in the same path under the newly created folder 'Results'")




