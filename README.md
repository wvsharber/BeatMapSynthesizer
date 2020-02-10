![alt text](https://github.com/wvsharber/BeatMapSynthesizer/blob/master/beatMapSynth.png "Image credit: Jacob Joyce")

# BeatMapSynth

This repo houses an in-progress project for developing a generative machine learning tool that will automatically create new Beat Saber mappings given a chosen song. 

More details to come!

## Introduction
Beat Saber is a wildly successful virtual reality (VR) video game that has appeared in homes and arcades since its release in May 2018. The premise of the game is similar to other rhythm video games, such as Dance Dance Revolution or Guitar Hero, where the objective is to perform events in sync with music. In Beat Saber, the events are slicing through approaching blocks with two lightsabers (one in each hand) in time with upbeat, mostly electronic dance music. The blocks must be hit at the appropriate time and in the appropriate direction. Since this game is played in VR, the game is fully immersive and involves full body movement, including arm, wrist, and hand movements, ducking and dodging around obstacles, and even dancing if you’re inclined! (For an idea of what the game looks like while playing, view this [video](https://www.youtube.com/watch?v=c9hP7jbJTk0)). 

While the game has seen massive success, it is limited by the number of official songs and associated mappings (e.g., how the blocks appear in each song) that are released by the company. However, one of the exciting features of Beat Saber includes the ability to develop and play custom songs and mappings on your individual system. These customizations are developed by amateur users and are available for PC versions of the game and can greatly expand the play time. Figure 1 compares cumulative number of custom songs released by users versus the songs released by the official Beat Saber company.

![alt text](https://github.com/wvsharber/BeatMapSynthesizer/blob/master/Figure1_CumulativeSongsReleased.png "Figure 1")
__Figure 1.__ Cumulative number of songs released for play by the official Beat Saber company versus custom songs by individual users.

While custom songs greatly extend the potential play time of Beat Saber, these customizations can be difficult to produce for novice users since the tools provided by Beat Saber and other third parties are rudimentary and involves a lot of user time input, knowledge of music and music software, and some advanced usage of computers. Furthermore, although there are thousands of custom songs available, they vary greatly in their quality, are limited in which difficulty levels are available, and are threatened by music copyright issues. BeatMapSynth aims to automatically produce custom mappings when provided a song chosen and owned by the user. This tool increases a user’s ability to develop their own content and play songs that are better suited to their tastes.

## Modeling Process
### Data Acquisition
The 3rd party website [BeatSaver](https://beatsaver.com/) provides an interface for users to download new songs and custom mappings to play on their PC versions of the game. I used the website’s API to download songs and associated user-generated mappings for training my models, as well as data on the difficulty level of the song and the user-generated rating of the song. Although there are over 20,000 user generated mappings on BeatSaver, I utilized only mappings with over 70% rating, approximately 8,000 mappings in total.

### Data Preparation
Custom scripts were used to download and process the song and map files. The Python library `librosa` was used extensively in this project for music analysis. Song files were analyzed for beat timings and spectral features. These were aligned with the block placement features in the map files. Map files are JSON style files that record where, when, and what type of block appears during the course of the song. It also records obstacles (e.g., walls) and events (e.g., extra lighting features that appear in the background of the level), although these features have been ignored for now.

### Modeling
To date, two types of models have been implemented. The first, a completely random model, is used as a baseline model for comparison. It takes in a song and returns a map file with blocks placed randomly on each beat. The second model is a Hidden Markov Model (HMM), which is used in modeling sequential data.  This model was chosen in order to create mappings that follow the standard flow of movements usually seen in Beat Saber levels. I used the Python library `markovify` to develop an HMM with 5 hidden states for each difficulty level in two datasets: (1) maps rated over 90% and (2) maps rated over 70%. Block placement was translated into "words" and "sentences" that markovify could model.

I developed three implementations of the HMMs:
1. Base HMM - This implementation uses the HMM to place blocks on each beat of a song. Generally the flow is good, but there is little structure throughout the song and no variation in block placement rate.
2. Segmented HMM - In order to increase block pattern repetition with song structure repetition (e.g., verse/chorus structure), I implemented a Laplacian segmentation algorithm to find similar segments during the course of a song. The HMM maps across a segment, and then the same sequence will be repeated the next time the segment is encountered in a song.
3. Rate Modulated Segmented HMM - Building off of the Segmented HMM, this implementation attempts to add block placement rate variation by building in a probability of increasing the number of blocks per beat based on the relative 'loudness' of the song at a given beat. 

I also trained a Random Forest Chain Classifier model, yet the maps it generated were generally much worse than even the random model. This may be worth revisiting in the future, but at the moment the HMM models are considered the superior models.

### Evaluation
Each model has been preliminarily evaluated based on the overall 'playability' of the generated maps, 'smoothness' of block placement, appropriateness of block placement and rate based on difficulty level. Full user testing surveys are planned, and parties interested in participating should contact me for inclusion! 

## Deployment
In addition to this GitHub repo (which contains a lot of exploratory and development code), a standalone executable program will be available for download and use on Mac OSX and Windows systems. If you are comfortable setting up a Python environment with conda, you can clone this repo, install the conda environment in `environment.yml`, and run the `beatmapsynth.py` script from the command line. Full instructions are given in the _Installation and Use_ section.