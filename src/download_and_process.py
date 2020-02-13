import numpy as np
import pandas as pd
import librosa
import librosa.display
import json
import requests
import pickle
import matplotlib.pyplot as plt
from io import BytesIO, TextIOWrapper, StringIO
from zipfile import ZipFile
import os
import soundfile as sf
import shutil
import time

def download_song_and_map(key):
    """Downloads the zipped folder of song and mapping data from the beatsaber api. Extracts files to a 'temp' folder 
    in the local directory."""
    response = requests.get(f"https://beatsaver.com/api/download/key/{key}")
    if response.status_code == 200:
        content_as_file = BytesIO(response.content)
        zip_file = ZipFile(content_as_file)
        for x in zip_file.filelist:
            print(zip_file.extract(x.filename, path = 'temporary'))
        return response.status_code
    else:
        return print(f"API call failed at {key} with error code {response.status_code}")

def get_available_difficulties(metadata_record):
    """Gets the difficulty levels that are present for a song in a metadata record."""
    levels = []
    for key, value in metadata_record['metadata']['difficulties'].items():
        if value == True or value == 'True':
            levels.append(key)
    return levels

def beat_frames_and_chroma(song_path, bpm):

    #Load music file, estimate beat frames, and compute chromagram
    y, sr = librosa.load(f"./temporary/{song_path}")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                 sr=sr,
                                                 trim = False,
                                                 units = 'frames',
                                                 start_bpm = bpm)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)
    return tempo, beat_frames, chromagram

def notes_processing(mapfile):
    """This function extracts the notes list from the mapfile and transforms it into a DataFrame of note features
    at 16th beat time points."""
    notes = pd.DataFrame.from_dict(mapfile['_notes']).add_prefix('notes')
    wide = widen_notes(notes)
#     long = to_sixteenth_beat(wide)
    return wide

def widen_notes(notes):
    """This function takes a DataFrame containing all the notes (i.e., blocks) from a level.dat file and widens
    the DataFrame such that one time point has seperate columns for each type of block."""
    wide = None
    x = 0
    while x < len(notes['notes_type'].unique()):
        if x == 0:
            #Make separate dataframe for first note type and add a suffix for the column names
            notes_a = notes[notes['notes_type'] == notes['notes_type'].unique()[x]].reset_index()
            notes_a.drop('index', axis = 1, inplace=True)
            notes_a = notes_a.add_suffix(f"_{notes['notes_type'].unique()[x]}")
            notes_a['_time'] = notes_a[f"notes_time_{notes['notes_type'].unique()[x]}"]
            notes_a.drop(f"notes_time_{notes['notes_type'].unique()[x]}", axis = 1, inplace = True)
            #Do the process again for the second note type
            notes_b = notes[notes['notes_type'] == notes['notes_type'].unique()[x+1]].reset_index()
            notes_b.drop('index', axis = 1, inplace=True)
            notes_b = notes_b.add_suffix(f"_{notes['notes_type'].unique()[x+1]}")
            notes_b['_time'] = notes_b[f"notes_time_{notes['notes_type'].unique()[x+1]}"]
            notes_b.drop(f"notes_time_{notes['notes_type'].unique()[x+1]}", axis = 1, inplace = True)
            #Merge the two dataframes
            wide = pd.merge(notes_a, notes_b, on = '_time', how = 'outer', sort = True)
            x += 2
        else: 
            #Continue adding and merging until all note types have been merged
            notes_c = notes[notes['notes_type'] == notes['notes_type'].unique()[x]].reset_index()
            notes_c.drop('index', axis = 1, inplace=True)
            notes_c = notes_c.add_suffix(f"_{notes['notes_type'].unique()[x]}")
            notes_c['_time'] = notes_c[f"notes_time_{notes['notes_type'].unique()[x]}"]
            notes_c.drop(f"notes_time_{notes['notes_type'].unique()[x]}", axis = 1, inplace = True)
            wide = pd.merge(wide, notes_c, on = '_time', how = 'outer', sort = True)
            x += 1
    #Replace NaN with 999
    wide.fillna(999, inplace = True)
    #Coerce all columns except _time back to integer
    for column in wide.columns:
        if column != '_time':
            wide[column] = wide[column].astype(int)
    return wide

def download_and_process(metadata):
    """This is the master function that downloads the zipped folder with music, map, and info files. It extracts
    features from the files and makes a single record out of the features."""
    #Construct list of download keys from metadata
    key_list = []
    for x in metadata:
        key_list.append(x['key'])
    
    #For each dowload key in the metadata, download and process the zip folder
    for key in key_list:
        available_difficulties = get_available_difficulties(list(filter(lambda x: x['key'] == key, metadata))[0])
        print(f"{key}:", available_difficulties)
        code = download_song_and_map(key)
        if code != 200:
            continue
        else:
            try:
                #open music file and process
                with open('./temporary/info.dat', 'rb') as i:
                    info = json.load(i)
                music_path = info['_songFilename']
                bpm = info['_beatsPerMinute']
                tempo, beat_frames, chromagram = beat_frames_and_chroma(music_path, bpm)
                
                #open map files and process
                for difficulty in available_difficulties:
                    map_file = open_map_file(difficulty)
                    notes_df = notes_processing(map_file)                    
                    notes_df['_time'] = round(notes_df['_time'], 3)
                    music_df = beat_number_and_chroma_v2(beat_frames, chromagram, notes_df['_time'])
                    df = pd.merge(music_df, notes_df, on = '_time', how = 'outer', sort = True)
                    df.iloc[:, 1:13] = df.iloc[:, 1:13].fillna(method = 'bfill', axis = 0)
                    df.iloc[:, 13:] = df.iloc[:, 13:].fillna(value = 999, axis = 0)
                    with open(f"../data/processed_data/{key}_{difficulty}.pkl", 'wb') as f:
                        pickle.dump(df, f)
                    
            except Exception as err:
                 print(f"{key}: \n {err}")
            finally:
                #delete temp directory
                shutil.rmtree('./temporary/')

def open_map_file(difficulty):
    """This function opens the map file listed in the info.dat file for the specificed difficulty level."""
    with open('./temporary/info.dat', 'rb') as i:
        info = json.load(i)
    for x in info['_difficultyBeatmapSets']:
        if x['_beatmapCharacteristicName'].casefold() == 'Standard'.casefold():
            for y in x['_difficultyBeatmaps']:
                if y['_difficulty'].casefold() == difficulty.casefold():
                    file_name = y['_beatmapFilename']
                    with open(f"./temporary/{file_name}", 'rb') as f:
                        map_file = json.load(f)
                    return map_file