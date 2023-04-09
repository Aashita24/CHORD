
import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.utils import make_chunks
import keras.backend as K
import librosa.display
import cv2
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import string
import random
from keras.applications import VGG16
import os
import scipy



# Load the tune recognition model
model = tf.keras.models.load_model('embdmodel_1.hdf5')
embedding_model=model.layers[2]

# Define function to preprocess input audio
#convert song to mel spectogram as siamese network doesn't work on sound directly
def create_spectrogram(clip,sample_rate,save_path):
    plt.interactive(False)
    fig=plt.figure(figsize=[0.72,0.72])
    S=librosa.feature.melspectrogram(y=clip,sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
    fig.savefig(save_path,dpi=400,bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del save_path,clip,sample_rate,fig,S
    
def load_img(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(150,150))
    return img


import pickle
with open('dict.pickle', 'rb') as handle:
    songspecdict = pickle.load(handle)



import streamlit as st
from audio_recorder_streamlit import audio_recorder
# Define Streamlit interface
st.set_page_config(page_title='Tune Recognition', page_icon=':musical_note:')
st.title('Tune Recognition with Microphone')
st.markdown('Click the button below to start recording your tune.')
recording = st.button('Start Recording')
if recording:
    # Start recording audio from device's microphone
    fs = 44100 # Sample rate
    duration = 10 # Duration in seconds
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    st.write('Recording...')

    # Wait for recording to finish
    sd.wait()
    # Apply pre-emphasis filter
    preemphasis_coeff = 0.07
    preemphasis_filter = np.array([1, -preemphasis_coeff])
    recording = scipy.signal.lfilter(preemphasis_filter, [1], recording.ravel())
    st.write('Recording stopped.')

    # Save the recorded audio to a temporary file
    rec_file = 'recording0.wav'
    write(rec_file, fs, recording)

    # Load the song to match
    song, sr = librosa.load(rec_file)
    to_match = np.copy(song[0:220500])

    # Create spectrogram image of the song to match
    create_spectrogram(to_match, sr, 'test.png')

    # Load the spectrogram image of the song to match
    to_match_img = load_img('test.png')
    to_match_img = np.expand_dims(to_match_img, axis=0)

    # Get the embedding of the song to match
    to_match_emb = embedding_model.predict(to_match_img)

   # Calculate the distances between the song to match and the songs in the database
    songsdistdict = {}
    for key, values in songspecdict.items():
        dist_array = []
        for embd in values:
            dist_array.append(np.linalg.norm(to_match_emb - embd))
            
        songsdistdict[key] = min(dist_array)
    song_titles=list(songsdistdict.keys())
    distances=list(songsdistdict.values())

    # Get the title and artist of the recognized song
    recognized_song_artist, recognized_song_title = song_titles[distances.index(min(distances))].split('-')
    recognized_song_title = os.path.splitext(recognized_song_title)[0]
    st.write(f'Artist: {recognized_song_artist}')
    st.write(f'Title: {recognized_song_title}')

    ##from musixmatch.api import Musixmatch
    from musixmatch import Musixmatch


    # Initialize Musixmatch API
    musixmatch = Musixmatch(apikey='2b0d0615efa782e95598a0e99bda4a60')

    # Search for the recognized song
    track_search_results = musixmatch.track_search(q_track=recognized_song_title, q_artist=recognized_song_artist, page_size=1, page=1, s_track_rating='desc')

    if track_search_results['message']['header']['status_code'] == 200:
        # Get the track ID for the top result
        track_id = track_search_results['message']['body']['track_list'][0]['track']['track_id']

        # Get the lyrics for the recognized song
        lyrics_result = musixmatch.track_lyrics_get(track_id=track_id)

        if lyrics_result['message']['header']['status_code'] == 200:
            # Get the lyrics
            lyrics = lyrics_result['message']['body']['lyrics']['lyrics_body']
            # Remove the annotation tags from the lyrics
            lyrics = lyrics.replace('******* This Lyrics is NOT for Commercial use *******', '').strip()
            st.write("Lyrics:\n", lyrics)
    else:
        st.write("Couldn't find lyrics for the recognized song.")  


    # Play the recognized song
    recognized_song_file = f'C:/Users/nisar/sem6 project/Siamese Network/seismese_net_songs/{song_titles[distances.index(min(distances))]}'
    recognized_song_audio, recognized_song_sr = librosa.load(recognized_song_file)

    audio_file = open(recognized_song_file, 'rb') # enter the filename with filepath
    audio_bytes = audio_file.read() # reading the file

    st.audio(audio_bytes, format='audio/wav', start_time=0) # displaying the audio
    
    





