from __future__ import print_function
import numpy as np
import pandas as pd
import librosa
import json
#import requests
import pickle
#import matplotlib.pyplot as plt
from io import BytesIO, TextIOWrapper, StringIO
from zipfile import ZipFile
import os
import soundfile as sf
import audioread
from pydub import AudioSegment
#from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
#from sklearn.ensemble import RandomForestClassifier
import markovify
import sklearn.cluster
#import librosa.display
import scipy
import sys
import argparse
import shutil

#Main Function:
def beat_map_synthesizer(song_path, song_name, difficulty, model, k=5, version = 2):
    """
    Function to load a music file and generate a custom Beat Saber map based on the specified model and difficulty. Outputs a zipped folder of necessary files to play the custom map in the Beat Saber game.
    
    ***
    song_path = string file path to music file location
    
    song_name = string to name level as it will appear in the game
    
    difficulty = desired difficulty level, can be: 'easy', 'normal', 'hard', 'expert', or 'expertPlus'
    
    model = desired model to use for map generation, can be: 'random', 'HMM', 'segmented_HMM', or 'rate_modulated_segmented_HMM'
    
    k = number of song segments if using a segmented model. Default is 5, may want to increase or decrease based on song complexity
   
    version = for HMM models, can choose either 1 or 2. 1 was trained on a smaller, but potentially higher quality dataset (custom maps with over 90% rating on beatsaver.com), while 2 was trained on a larger dataset of custom maps with over 70% rating, so it may have a larger pool of "potential moves."
    ***
    """
    if model == 'random':
        random_mapper(song_path, song_name, difficulty)
    elif model == 'HMM':
        HMM_mapper(song_path, song_name, difficulty, version = version)
    elif model == 'segmented_HMM':
        segmented_HMM_mapper(song_path, song_name, difficulty, k = k, version = version)
    elif model == 'rate_modulated_segmented_HMM':
        rate_modulated_segmented_HMM_mapper(song_path, song_name, difficulty, version = version, k = k)
    else:
        print('Please specify model for mapping.')

#Basic File Writing Functions
def write_info(song_name, bpm, difficulty):
    """This function creates the 'info.dat' file that needs to be included in the custom folder."""

    difficulty_rank = None
    jump_movement = None
    if difficulty.casefold() == 'easy'.casefold():
        difficulty_rank = 1
        jump_movement = 8
        diff_name = 'Easy'
    elif difficulty.casefold() == 'normal'.casefold():
        difficulty_rank = 3
        jump_movement = 10
        diff_name = 'Normal'
    elif difficulty.casefold() == 'hard'.casefold():
        difficulty_rank = 5
        jump_movement = 12
        diff_name = 'Hard'
    elif difficulty.casefold() == 'expert'.casefold():
        difficulty_rank = 7
        jump_movement = 14
        diff_name = 'Expert'
    elif difficulty.casefold() == 'expertPlus'.casefold():
        difficulty_rank = 9
        jump_movement = 16
        diff_name = 'ExpertPlus'
            
    info = {'_version': '2.0.0',
            '_songName': f"{song_name}",
            '_songSubName': '',
            '_songAuthorName': '',
            '_levelAuthorName': 'BeatMapSynth',
            '_beatsPerMinute': round(bpm),
            '_songTimeOffset': 0,
            '_shuffle': 0,
            '_shufflePeriod': 0,
            '_previewStartTime': 10,
            '_previewDuration': 30,
            '_songFilename': 'song.egg',
            '_coverImageFilename': 'cover.jpg',
            '_environmentName': 'DefaultEnvironment',
            '_customData': {},
             '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': 'Standard',
                                         '_difficultyBeatmaps': [{'_difficulty': diff_name,
                                                                  '_difficultyRank': difficulty_rank,
                                                                  '_beatmapFilename': f"{difficulty}.dat",
                                                                  '_noteJumpMovementSpeed': jump_movement,
                                                                  '_noteJumpStartBeatOffset': 0,
                                                                  '_customData': {}}]}]}
    with open('info.dat', 'w') as f:
        json.dump(info, f)

def write_level(difficulty, events_list, notes_list, obstacles_list):
    """This function creates the 'level.dat' file that contains all the data for that paticular difficulty level"""
    
    level = {'_version': '2.0.0',
             '_customData': {'_time': '', #not sure what time refers to 
                             '_BPMChanges': [], 
                             '_bookmarks': []},
             '_events': events_list,
             '_notes': notes_list,
             '_obstacles': obstacles_list}
    with open(f"{difficulty}.dat", 'w') as f:
        json.dump(level, f)

def music_file_converter(song_path):
    """This function makes sure the file type of the provided song will be converted to the music file type that 
    Beat Saber accepts"""
    if song_path.endswith('.mp3'):
        AudioSegment.from_mp3(song_path).export('song.egg', format='ogg')
    elif song_path.endswith('.wav'):
        AudioSegment.from_wav(song_path).export('song.egg', format='ogg')
    elif song_path.endswith('.flv'):
        AudioSegment.from_flv(song_path).export('song.egg', format='ogg')
    elif song_path.endswith('.raw'):
        AudioSegment.from_raw(song_path).export('song.egg', format='ogg')
    elif song_path.endswith('.ogg') or song_path.endswith('.egg'):
        shutil.copy2(song_path, 'song.egg')
    else:
        print("Unsupported song file type. Choose a file of type .mp3, .wav, .flv, .raw, or .ogg.")

def events_writer(beat_times):
    """Placeholder function for writing a list of events to be incorporated into a beatmap file. May have future support."""
    events_list = []
    return events_list

def obstacles_writer(beat_times, difficulty):
    """Placeholder function for writing a list of obstacles to be incorporated into a beatmap file."""
    obstacles_list = []
    return obstacles_list

def zip_folder_exporter(song_name, difficulty):
    "This function exports the zip folder containing the info.dat, difficulty.dat, cover.jpg, and song.egg files."
    files = ['info.dat', f"{difficulty}.dat", 'cover.jpg', 'song.egg']
    with ZipFile(f"{song_name}.zip", 'w') as custom:
        for file in files:
            custom.write(file)
    for file in files:
        if file != 'cover.jpg':
            os.remove(file)

#Random Mapping Functions
def random_mapper(song_path, song_name, difficulty):
    """Function to output the automatically created completely random map (i.e. baseline model) for a provided song. Returns a zipped folder that can be unzipped and placed in the 'CustomMusic' folder in the Beat Saber game directory and played. CAUTION: This is completely random and is likely not enjoyable if even playable!"""
    #Load song and get beat features
    print("Loading Song...")
    bpm, beat_times, y, sr = beat_features(song_path)
    print("Song loaded successfully!")
    #Write lists for note placement, event placement, and obstacle placement
    print("Random mapping...")
    #notes_list = random_notes_writer(beat_times, difficulty) 
    notes_list = random_notes_writer_v2(beat_times, difficulty, bpm) #fixes _time != beat time
    events_list = events_writer(beat_times)
    obstacles_list = obstacles_writer(beat_times, difficulty)
    print("Mapping done!")
    #Write and zip files
    print("Writing files to disk...")
    write_info(song_name, bpm, difficulty)
    write_level(difficulty, events_list, notes_list, obstacles_list)
    print("Converting music file...")
    music_file_converter(song_path)
    print("Zipping folder...")
    zip_folder_exporter(song_name, difficulty)
    print("Finished! Look for zipped folder in your current path, unzip the folder, and place in the 'CustomMusic' folder in the Beat Saber directory")

def beat_features(song_path):
    """This function takes in the song stored at 'song_path' and estimates the bpm and beat times."""
    #Load song and split into harmonic and percussive parts.
    y, sr = librosa.load(song_path)
    #y_harmonic, y_percussive = librosa.effects.hpss(y)
    #Isolate beats and beat times
    bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim = False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return bpm, beat_times, y, sr

def random_notes_writer_v2(beat_times, difficulty, bpm):
    """This function randomly places blocks at approximately each beat or every other beat depending on the difficulty."""
    notes_list = []
    line_index = [0, 1, 2, 3]
    line_layer = [0, 1, 2]
    types = [0, 1, 2, 3]
    directions = list(range(0, 10))
    #beat_times = [float(x) for x in beat_times]
    beat_times = [x*(bpm/60) for x in beat_times] #list(range(len(beat_times)))
    
    if difficulty == 'Easy' or difficulty == 'Normal':
        for beat in beat_times:
            empty = np.random.choice([0,1])
            if empty == 1:
                note = {'_time': beat,
                        '_lineIndex': int(np.random.choice(line_index)),
                        '_lineLayer': int(np.random.choice(line_layer)),
                        '_type': int(np.random.choice(types)),
                        '_cutDirection': int(np.random.choice(directions))}
                notes_list.append(note)
            else:
                continue
    else:
        random_beats = np.random.choice(beat_times, np.random.choice(range(len(beat_times)))) #randomly choose beats to have more than one note placed
        randomly_duplicated_beat_times = np.concatenate([beat_times, random_beats])
        randomly_duplicated_beat_times.sort()
        randomly_duplicated_beat_times = [float(x) for x in randomly_duplicated_beat_times]
        for beat in randomly_duplicated_beat_times:
            note = {'_time': beat,
                    '_lineIndex': int(np.random.choice(line_index)),
                    '_lineLayer': int(np.random.choice(line_layer)),
                    '_type': int(np.random.choice(types)),
                    '_cutDirection': int(np.random.choice(directions))}
            notes_list.append(note)
    #Remove potential notes that come too early in the song:
    for i, x in enumerate(notes_list):
        if notes_list[i]['_time'] >= 0 and notes_list[i]['_time'] <= 1.5:
            del notes_list[i]
        elif notes_list[i]['_time'] > beat_times[-1]:
            del notes_list[i]

    return notes_list

#Hidden Markov Models Mapping Functions
def HMM_mapper(song_path, song_name, difficulty, version = 2):
    """This function generates a custom map based on a Hidden Markov Model."""
    #Load song and get beat features
    print("Loading Song...")
    bpm, beat_times, y, sr = beat_features(song_path)
    beat_times = [x*(bpm/60) for x in beat_times] #list(range(len(beat_times)))
    print("Song loaded successfully!")
    #Write lists for note placement, event placement, and obstacle placement
    print("Mapping with Hidden Markov Model...")
    notes_list = HMM_notes_writer(beat_times, difficulty, version)
    events_list = events_writer(beat_times)
    obstacles_list = obstacles_writer(beat_times, difficulty)
    print("Mapping done!")
    #Write and zip files
    print("Writing files to disk...")
    write_info(song_name, bpm, difficulty)
    write_level(difficulty, events_list, notes_list, obstacles_list)
    print("Converting music file...")
    music_file_converter(song_path)
    print("Zipping folder...")
    zip_folder_exporter(song_name, difficulty)
    print("Finished! Look for zipped folder in your current path, unzip the folder, and place in the 'CustomMusic' folder in the Beat Saber directory")

def walk_to_df(walk):
    """Function for turning a Markov walk sequence into a DataFrame of note placement predictions"""
    sequence = []
    for step in walk:
        sequence.append(step.split(","))
    constant = ['notes_type_0', 'notes_lineIndex_0', 'notes_lineLayer_0',
                    'notes_cutDirection_0', 'notes_type_1', 'notes_lineIndex_1', 'notes_lineLayer_1', 
                    'notes_cutDirection_1', 'notes_type_3', 'notes_lineIndex_3',
                    'notes_lineLayer_3', 'notes_cutDirection_3']
    df = pd.DataFrame(sequence, columns = constant)
    return df

def HMM_notes_writer(beat_list, difficulty, version):
    """Writes a list of notes based on a Hidden Markov Model walk."""
    #Load model
    if version == 1:
        with open(f"./models/HMM_{difficulty}.pkl", 'rb') as m:
            MC = pickle.load(m)
    elif version == 2:
        with open(f"./models/HMM_{difficulty}_v2.pkl", 'rb') as m:
            MC = pickle.load(m)
    #Set note placement rate dependent on difficulty level
    counter = 2
    beats = []
    rate = None
    if difficulty == 'easy':
        rate = 3
    elif difficulty == 'normal':
        rate = 2
    else:
        rate = 1
    while counter <= len(beat_list):
        beats.append(counter)
        counter += rate
    #Get HMM walk long enough to cover number of beats
    random_walk = MC.walk()
    while len(random_walk) < len(beats):
        random_walk = MC.walk()
    df_walk = walk_to_df(random_walk)
    #Combine beat numbers with HMM walk steps
    df_preds = pd.concat([pd.DataFrame(beats, columns = ['_time']), df_walk], axis = 1, sort = True)
    df_preds.dropna(axis = 0, inplace = True)
    #Write notes dictionaries
    notes_list = []
    for index, row in df_preds.iterrows():
        for x in list(filter(lambda y: y.startswith('notes_type'), df_preds.columns)):
            if row[x] != '999':
                num = x[-1]
                note = {'_time': row['_time'],
                        '_lineIndex': int(row[f"notes_lineIndex_{num}"]),
                        '_lineLayer': int(row[f"notes_lineLayer_{num}"]),
                        '_type': int(num),
                        '_cutDirection': int(row[f"notes_cutDirection_{num}"])}
                notes_list.append(note)
   #Remove potential notes that come too early in the song:
    for i, x in enumerate(notes_list):
        if notes_list[i]['_time'] >= 0 and notes_list[i]['_time'] <= 1.5:
            del notes_list[i]
        elif notes_list[i]['_time'] > beats[-1]:
            del notes_list[i]

    return notes_list

#Segmented HMM Functions

def segmented_HMM_mapper(song_path, song_name, difficulty, k = 5, version = 2):
    """This function generates a custom map based on a HMM model that operates on song segments. First, Laplacian song segmentation is performed to identify similar portions of songs. Then, a HMM is used to generate a block sequence through the first of each of these identified song segments. If that segment is repeated later in the song, the block sequence will be repeated."""
    #Load song and get beat features
    print("Loading Song...")
    bpm, beat_times, y, sr = beat_features(song_path)
    beat_times = [x*bpm/60 for x in beat_times]
    print("Song loaded successfully!")
    #Write lists for note placement, event placement, and obstacle placement
    print("Mapping with segmented Hidden Markov Model...")
    notes_list = segmented_HMM_notes_writer(y, sr, k, difficulty, version)
    events_list = events_writer(beat_times)
    obstacles_list = obstacles_writer(beat_times, difficulty)
    print("Mapping done!")
    #Write and zip files
    print("Writing files to disk...")
    write_info(song_name, bpm, difficulty)
    write_level(difficulty, events_list, notes_list, obstacles_list)
    print("Converting music file...")
    music_file_converter(song_path)
    print("Zipping folder...")
    zip_folder_exporter(song_name, difficulty)
    print("Finished! Look for zipped folder in your current path, unzip the folder, and place in the 'CustomMusic' folder in the Beat Saber directory")

def laplacian_segmentation(y, sr, k = 5):
    """This function uses the Laplacian Segmentation method described in McFee and Ellis, 2014, and adapted from example code in the librosa documentation. It returns the segment boundaries (in frame number and time and segment ID's of isolated music file segments."""
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                        bins_per_octave=BINS_PER_OCTAVE,
                                        n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                        ref=np.max)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # For plotting purposes, we'll need the timing of the beats
    # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                x_min=0,
                                                                x_max=C.shape[1]),
                                                                sr=sr)
    
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                          sym=True)
    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)
    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5
    # If we want k clusters, use the first k normalized eigenvectors.
    # Fun exercise: see how the segmentation changes as you vary k
    k = k
    X = evecs[:, :k] / Cnorm[:, k-1:k]
    KM = sklearn.cluster.KMeans(n_clusters=k)
    seg_ids = KM.fit_predict(X)
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
    # Compute the segment label for each boundary
    bound_segs = list(seg_ids[bound_beats])
    # Convert beat indices to frames
    bound_frames = beats[bound_beats]
    # Make sure we cover to the end of the track
    bound_frames = librosa.util.fix_frames(bound_frames,
                                           x_min=None,
                                           x_max=C.shape[1]-1)
    bound_times = librosa.frames_to_time(bound_frames)
    bound_times = [(x/60)*tempo for x in bound_times]
    beat_numbers = list(range(len(bound_frames)))
    bound_beats = np.append(bound_beats, list(range(len(beats)))[-1])
    segments = list(zip(zip(bound_times, bound_times[1:]), zip(bound_beats, bound_beats[1:]), bound_segs))
    
    return segments, beat_times, tempo

def segments_to_df(segments):
    """Helper function to translate a song semgmenation to a pandas DataFrame."""
    lengths = []
    for seg in segments:
        length = seg[1][1] - seg[1][0]
        lengths.append(length)
    df = pd.concat([pd.Series(lengths, name = 'length'), pd.Series([x[2] for x in segments], name = 'seg_no')], axis = 1)
    return df

def segment_predictions(segment_df, HMM_model):
    """This function predicts a Markov chain walk for each segment of a segmented music file. It will repeat a walk for segments that it has already mapped previously (truncating or extending as necessary)."""
   
    preds = pd.DataFrame([])
    completed_segments = {}
    for index, row in segment_df.iterrows():
        if row['seg_no'] not in completed_segments.keys():
            if index == 0:
                pred = HMM_model.walk()
                while len(pred) < row['length']:
                    pred = HMM_model.walk()
                completed_segments.update({row['seg_no']: {'start':0, 'end': len(pred)}})
                preds = pd.concat([preds, pd.Series(pred[0: row['length']])], axis = 0, ignore_index = True)
                
            else:
                try:
                    pred = HMM_model.walk(init_state = tuple(preds.iloc[-5:, 0]))
                    while len(pred) < row['length']:
                        pred = HMM_model.walk(init_state = tuple(preds.iloc[-5:, 0]))
                    completed_segments.update({row['seg_no']: {'start': len(preds)+1, 'end': len(preds)+len(pred)}})
                    preds = pd.concat([preds, pd.Series(pred[0: row['length']])], axis = 0, ignore_index = True)
                except:
                    pred = HMM_model.walk()
                    while len(pred) < row['length']:
                        pred = HMM_model.walk()
                    completed_segments.update({row['seg_no']: {'start': len(preds)+1, 'end': len(preds)+len(pred)}})
                    preds = pd.concat([preds, pd.Series(pred[0: row['length']])], axis = 0, ignore_index = True)

        else:
            if row['length'] <= (completed_segments[row['seg_no']]['end'] - completed_segments[row['seg_no']]['start']): 
                pred = preds.iloc[completed_segments[row['seg_no']]['start']: completed_segments[row['seg_no']]['start'] + row['length'], 0]
                preds = pd.concat([preds, pred], axis = 0, ignore_index = True)
            else:
                try:
                    extend = HMM_model.walk(init_state = tuple(preds.iloc[completed_segments[row['seg_no']]['end'] - 5 : completed_segments[row['seg_no']]['end'], 0]))
                    pred = preds.iloc[completed_segments[row['seg_no']]['start']: completed_segments[row['seg_no']]['end'], 0]
                    diff = row['length'] - len(pred)
                    pred = pd.concat([pred, pd.Series(extend[0: diff+1])], axis = 0, ignore_index = True)
                    completed_segments.update({row['seg_no']: {'start': len(preds)+1, 'end': len(preds)+len(pred)}})
                    preds = pd.concat([preds, pred], axis = 0, ignore_index = True)
                except:
                    extend = HMM_model.walk()
                    pred = preds.iloc[completed_segments[row['seg_no']]['start']: completed_segments[row['seg_no']]['end'], 0]
                    diff = row['length'] - len(pred)
                    pred = pd.concat([pred, pd.Series(extend[0: diff+1])], axis = 0, ignore_index = True)
                    completed_segments.update({row['seg_no']: {'start': len(preds)+1, 'end': len(preds)+len(pred)}})
                    preds = pd.concat([preds, pred], axis = 0, ignore_index = True)
    
    preds_list = list(preds.iloc[:, 0])
    preds = walk_to_df(preds_list)
    return preds

def segmented_HMM_notes_writer(y, sr, k, difficulty, version = 2):
    """This function writes the list of notes based on the segmented HMM model."""
    #Load model:
    if version == 1:
        with open(f"./models/HMM_{difficulty}.pkl", 'rb') as m:
            MC = pickle.load(m)
    elif version == 2:
        with open(f"./models/HMM_{difficulty}_v2.pkl", 'rb') as m:
            MC = pickle.load(m)
            
    segments, beat_times, tempo = laplacian_segmentation(y, sr, k)
    segments_df = segments_to_df(segments)
    preds = segment_predictions(segments_df, MC)
    #Combine beat numbers with HMM walk steps
    beats = [(x/60)* tempo for x in beat_times]
    df_preds = pd.concat([pd.DataFrame(beats, columns = ['_time']), preds], axis = 1, sort = True)
    df_preds.dropna(axis = 0, inplace = True)
    #Write notes dictionaries
    notes_list = []
    for index, row in df_preds.iterrows():
        for x in list(filter(lambda y: y.startswith('notes_type'), df_preds.columns)):
            if row[x] != '999':
                num = x[-1]
                note = {'_time': row['_time'],
                        '_lineIndex': int(row[f"notes_lineIndex_{num}"]),
                        '_lineLayer': int(row[f"notes_lineLayer_{num}"]),
                        '_type': int(num),
                        '_cutDirection': int(row[f"notes_cutDirection_{num}"])}
                notes_list.append(note)
    #Remove potential notes that come too early in the song:
    for i, x in enumerate(notes_list):
        if notes_list[i]['_time'] >= 0 and notes_list[i]['_time'] <= 1.5:
            del notes_list[i]
        elif notes_list[i]['_time'] > beats[-1]:
            del notes_list[i]
    
    return notes_list

#Rate Modulated Segmented HMM mapping functions
def rate_modulated_segmented_HMM_mapper(song_path, song_name, difficulty, version = 2, k = 5):
    """This function generates the files for a custom map using a rate modulated segmented HMM model."""
    #Load song and get beat features
    print("Loading Song...")
    bpm, beat_times, y, sr = beat_features(song_path)
    print("Song loaded successfully!")
    #Write lists for note placement, event placement, and obstacle placement
    print("Mapping with rate modulated segmented Hidden Markov Model...")
    notes_list, modulated_beat_list = rate_modulated_segmented_HMM_notes_writer(y, sr, k, difficulty, version)
    events_list = events_writer(modulated_beat_list)
    obstacles_list = obstacles_writer(modulated_beat_list, difficulty)
    print("Mapping done!")
    #Write and zip files
    print("Writing files to disk...")
    write_info(song_name, bpm, difficulty)
    write_level(difficulty, events_list, notes_list, obstacles_list)
    print("Converting music file...")
    music_file_converter(song_path)
    print("Zipping folder...")
    zip_folder_exporter(song_name, difficulty)
    print("Finished! Look for zipped folder in your current path, unzip the folder, and place in the 'CustomMusic' folder in the Beat Saber directory")

def choose_rate(db, difficulty):
    """
    This function modulates the block placement rate by using the average amplitude (i.e., 'loudness') across beats to choose how many blocks per beat will be placed. Takes in the difficulty level and the amplitude and returns an integer in the set {0, 1, 2, 4, 8, 16}.
    
    If you are finding that your maps are too fast or too slow for you, you might want to play with the probabilities in this file.
    """
    db = np.abs(db)
    p = None
    if difficulty.casefold() == 'easy'.casefold():
        if db > 70:
            p = [0.95, 0.05, 0, 0, 0, 0]
        elif db <= 70 and db > 55:
            p = [0.90, 0.10, 0, 0, 0, 0]
        elif db <= 55 and db > 45:
            p = [0.80, 0.2, 0, 0, 0, 0]
        elif db <= 45 and db > 35:
            p = [0.4, 0.5, 0.1, 0, 0, 0]
        else:
            p = [0.3, 0.6, 0.1, 0, 0, 0]
    elif difficulty.casefold() == 'normal'.casefold():
        if db > 70:
            p = [0.95, 0.05, 0, 0, 0, 0]
        elif db <= 70 and db > 55:
            p = [0.5, 0.5, 0, 0, 0, 0]
        elif db <= 55 and db > 45:
            p = [0.3, 0.7, 0, 0, 0, 0]
        elif db <= 45 and db > 35:
            p = [0.2, 0.7, 0.1, 0, 0, 0]
        else:
            p = [0.05, 0.7, 0.25, 0, 0, 0]
    elif difficulty.casefold() == 'hard'.casefold():
        if db > 70:
            p = [0.95, 0.05, 0, 0, 0, 0]
        elif db <= 70 and db > 55:
            p = [0.5, 0.5, 0, 0, 0, 0]
        elif db <= 55 and db > 45:
            p = [0.2, 0.6, 0.2, 0, 0, 0]
        elif db <= 45 and db > 35:
            p = [0.1, 0.5, 0.4, 0, 0, 0]
        else:
            p = [0.05, 0.35, 0.6, 0, 0, 0]
    elif difficulty.casefold() == 'expert'.casefold():
        if db > 70:
            p = [0.8, 0.2, 0, 0, 0, 0]
        elif db <= 70 and db > 55:
            p = [0.2, 0.7, 0.1, 0, 0, 0]
        elif db <= 55 and db > 50:
            p = [0.1, 0.4, 0.3, 0.2, 0, 0]
        elif db <= 50 and db > 45:
            p = [0, 0.05, 0.6, 0.35, 0, 0]
        else:
            p = [0, 0, 0.35, 0.65, 0, 0]
    elif difficulty.casefold() == 'expertPlus'.casefold():
        if db > 70:
            p = [0, 0.5, 0.4, 0.1, 0, 0]
        elif db <= 70 and db > 55:
            p = [0, 0.3, 0.6, 0.1, 0, 0]
        elif db <= 55 and db > 50:
            p = [0, 0.1, 0.6, 0.3, 0, 0]
        elif db <= 50 and db > 45:
            p = [0, 0.05, 0.1, 0.6, 0.25, 0]
        else:
            p = [0, 0, 0, 0.5, 0.3, 0.2]
    return np.random.choice([0, 1, 2, 4, 8, 16], p = p)

def amplitude_rate_modulation(y, sr, difficulty):
    """This function uses the average amplitude (i.e., 'loudness') of a beat and the difficulty level to determine 
    how many blocks will be placed within the beat. Returns a list of beat numbers."""
    #Make amplitude matrix
    D = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(D, ref=np.max)
    #Get beat frames and sync with amplitudes
    tempo, beat_frames = librosa.beat.beat_track(y, sr, trim = False)
    beat_db = pd.DataFrame(librosa.util.sync(db, beat_frames, aggregate = np.mean))
    #Mean amplitude per beat
    avg_beat_db = beat_db.mean()
    #Choose rates and smooth rate transitions
    rates = [0]
    counter = 1
    while counter < len(avg_beat_db)-1:
        rate = choose_rate(np.mean([avg_beat_db.iloc[counter-1], avg_beat_db.iloc[counter], avg_beat_db.iloc[counter+1]]), difficulty)
        diff = np.abs(rate - rates[-1])
        if difficulty.casefold() == 'expert'.casefold():
            maxdiff = 4
        elif difficulty.casefold() == 'expertPlus'.casefold():
            maxdiff = 8
        else:
            maxdiff = 2
        while diff > maxdiff:
            rate = choose_rate(np.mean([avg_beat_db.iloc[counter-1], avg_beat_db.iloc[counter], avg_beat_db.iloc[counter+1]]), difficulty)
            diff = rates[-1] - rate
        if rate == 4 and rates[-1] == 4: #and rates[-2] == 4:
            rate = np.random.choice([0, 1, 2])
        rates.append(rate)
        counter +=1
    #Make list of beat numbers based on rates
    beat_num_list = []
    for ind, val in enumerate(rates):
        if val == 0:
            continue
        elif val == 1:
            beat_num_list.append(ind)
        else:
            num_list = [ind, ind+1]
            for x in range(1, val):
                num_list.append(ind+(x/val))
            for y in num_list:
                beat_num_list.append(y)
    beat_num_list = list(set(beat_num_list))
    beat_num_list.sort()
    return beat_num_list

def segments_to_df_rate_modulated(segments, modulated_beat_list):
    """This function returns a DataFrame of the number of blocks needed for each song segment."""
    expanded_beat_list = []
    for x in segments:
        for y in modulated_beat_list:
            if y > x[1][0] and y <= x[1][1]:
                expanded_beat_list.append({'_time': y, 'segment': x[2]})
    
    df = pd.DataFrame([], columns = ['length', 'seg_no'])
    counter = 0
    first = None
    last = None
    while counter < len(expanded_beat_list):
        if counter == 0:
            first = counter
            counter += 1
        elif expanded_beat_list[counter]['segment'] != expanded_beat_list[counter-1]['segment']:
            first = counter
            counter += 1
        elif expanded_beat_list[counter] == expanded_beat_list[-1]:
            length = len(expanded_beat_list[first: -1])
            df = df.append(pd.DataFrame({'length': length, 'seg_no': expanded_beat_list[-1]['segment']}, index = [0]))
            break
        elif expanded_beat_list[counter]['segment'] == expanded_beat_list[counter+1]['segment']:
            counter += 1  
        elif expanded_beat_list[counter]['segment'] != expanded_beat_list[counter+1]['segment']:
            last = counter
            length = len(expanded_beat_list[first: last+1])
            df = df.append(pd.DataFrame({'length': length, 'seg_no': expanded_beat_list[counter]['segment']}, index = [0]))
            counter += 1
    
    return df

def rate_modulated_segmented_HMM_notes_writer(y, sr, k, difficulty, version):
    """Function to write the notes to a list after predicting with the rate modulated segmented HMM model."""
    #Load model:
    if version == 1:
        with open(f"./models/HMM_{difficulty}.pkl", 'rb') as m:
            MC = pickle.load(m)
    elif version == 2:
        with open(f"./models/HMM_{difficulty}_v2.pkl", 'rb') as m:
            MC = pickle.load(m)
    
    segments, beat_times, bpm = laplacian_segmentation(y, sr, k)
    modulated_beat_list = amplitude_rate_modulation(y, sr, difficulty)
    segments_df = segments_to_df_rate_modulated(segments, modulated_beat_list)
    preds = segment_predictions(segments_df, MC)
    #Combine beat numbers with HMM walk steps
    beat_times = [(x/60)*bpm for x in beat_times]
    beat_count = list(range(len(beat_times)))
    beats = pd.concat([pd.Series(beat_times, name = '_time'), pd.Series(beat_count, name = 'beat_count')], axis = 1)
    for index, value in beats.iterrows():
        if value['beat_count'] not in modulated_beat_list:
            beats.drop(index = index, inplace=True)
    merged_beats = pd.merge(left = beats, right = pd.Series(modulated_beat_list, name = 'beat_count'), how='outer', on='beat_count', sort = True)
    merged_beats.interpolate(inplace=True)
    merged_beats.drop(columns = 'beat_count', inplace = True)
    
    df_preds = pd.concat([merged_beats, preds], axis = 1, sort = True)
    df_preds.dropna(axis = 0, inplace = True)
    #Write notes dictionaries
    notes_list = []
    for index, row in df_preds.iterrows():
        for x in list(filter(lambda y: y.startswith('notes_type'), df_preds.columns)):
            if row[x] != '999':
                num = x[-1]
                note = {'_time': row['_time'],
                        '_lineIndex': int(row[f"notes_lineIndex_{num}"]),
                        '_lineLayer': int(row[f"notes_lineLayer_{num}"]),
                        '_type': int(num),
                        '_cutDirection': int(row[f"notes_cutDirection_{num}"])}
                notes_list.append(note)
    #Remove potential notes that come too early in the song:
    for i, x in enumerate(notes_list):
        if notes_list[i]['_time'] >= 0 and notes_list[i]['_time'] <= 1.5:
            del notes_list[i]
        elif notes_list[i]['_time'] > beat_times[-1]:
            del notes_list[i]

    return notes_list, modulated_beat_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('song_path', metavar='path', type=str, help='File Path to song file')
    parser.add_argument('song_name', type=str, help='Name of song to be displayed in Beat Saber')
    parser.add_argument('difficulty', type=str, help="Desired difficulty level: 'easy', 'normal', 'hard', 'expert', or 'expertPlus'")
    parser.add_argument('model', type=str, help="Desired model for mapping: 'random', 'HMM', 'segmented_HMM', 'rate_modulated_segmented_HMM'")
    parser.add_argument('-k', type=int, help="Number of expected segments for segmented model. Default 5", default=5, required=False)
    parser.add_argument('--version', type=int, help="Version of HMM model to use: 1 (90% rating or greater) or 2 (70% rating or greater)", default=2, required=False)

    args = parser.parse_args()
    
    beat_map_synthesizer(args.song_path, args.song_name, args.difficulty, args.model, args.k, args.version)
    
