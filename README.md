![alt text](https://github.com/wvsharber/BeatMapSynthesizer/blob/master/beatMapSynth_Banner.jpg "Image credit: fellow Beat Saber enthusiast, Jacob Joyce")

# BeatMapSynth

__Instructions for installation and use are at the bottom of this README.__

You can read more about BeatMapSynth in my blog post [here](https://medium.com/swlh/beatmapsynth-an-automatic-song-mapper-for-beat-saber-aa9e59f093f8).

## Introduction
Beat Saber is a wildly successful virtual reality (VR) video game that has appeared in homes and arcades since its release in May 2018. The premise of the game is similar to other rhythm video games, such as Dance Dance Revolution or Guitar Hero, where the objective is to perform events in sync with music. In Beat Saber, the events are slicing through approaching blocks with two lightsabers (one in each hand) in time with upbeat, mostly electronic dance music. The blocks must be hit at the appropriate time and in the appropriate direction. Since this game is played in VR, the game is fully immersive and involves full body movement, including arm, wrist, and hand movements, ducking and dodging around obstacles, and even dancing if you’re inclined! (For an idea of what the game looks like while playing, view this [video](https://www.youtube.com/watch?v=c9hP7jbJTk0)). 

While the game has seen massive success, it is limited by the number of official songs and associated mappings (e.g., how the blocks appear in each song) that are released by the company. However, one of the exciting features of Beat Saber includes the ability to develop and play custom songs and mappings on your individual system. These customizations are developed by amateur users and are available for PC versions of the game and can greatly expand the play time. Figure 1 compares cumulative number of custom songs released by users versus the songs released by the official Beat Saber company.

![alt text](https://github.com/wvsharber/BeatMapSynthesizer/blob/master/reports/Figures/Figure1_CumulativeSongsReleased.png "Figure 1")
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

I also trained a Random Forest Chain Classifier model, yet the maps it generated were generally much worse than even the random model. This may be worth revisiting in the future, but at the moment the HMM models are considered the preferred models.

### Evaluation
Each model has been preliminarily evaluated based on the overall 'playability' of the generated maps, 'smoothness' of block placement, appropriateness of block placement and rate based on difficulty level. Full user testing surveys are planned, and parties interested in participating should fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLScYoJpZzT1v7UDY84rNbkPwubBIV5gKMlE_JtbvYXVZhtfgeQ/viewform?usp=sf_link) for inclusion! 

## Deployment
In addition to this GitHub repo (which contains a lot of exploratory and development code), a standalone executable program will be available for download and use on Mac OSX and Windows systems. If you are comfortable setting up a Python environment with conda, you can clone this repo, install the conda environment in `environment.yml`, and run the `beatmapsynth.py` script from the command line. Full instructions are given in the _Installation and Use_ section.

---

## _Installation and Use_ 

Current instructions for installation and use are limited to a command line option with a Python script. More user-friendly options are coming soon!

If you are familiar with conda environments and already have Python and conda installed on your system, clone this repo and set up the environment in the appropriate `environment.yml` file for your OS, activate the environment, and then proceed to step 4 in the instructions below to finish setup and run BeatMapSynth. 

If you do not already have Anaconda and Python, follow the instructions below to set them up.

1. Install Anaconda and Python on your system. Anaconda installs Python automatically and you can download the appropriate version [here](https://www.anaconda.com/products/individual). Follow the instructions to complete the installation. __Note:__ If you're working on a Windows machine, Anaconda will install the Anaconda Prompt, which is similar to Command Prompt. For the rest of these instructions, _make sure to use the Anaconda Prompt_, and not the native Command Prompt. For Mac users, use the Terminal for all of the instructions.

2. Clone this github repo onto your system. Take note of its location on your machine and then navigate into the directory from the command line (Anaconda Prompt on Windows or the Terminal in Mac OSX, e.g. `cd path_to_directory`)

3. Use the following command to set up the BeatMapSynth environment:

    For Windows: ```conda env create -f Windows_environment.yml```

    For Mac OSX: ```conda env create -f OSX_environment.yml```

    And then activate the environment:

    ```conda activate BSMapSynth-env```

4. Place the necessary models into the `models/` directory by downloading [this zipped folder](https://drive.google.com/open?id=1p7j0sENy0DzcMHd3iQ_LQw14j6OcKLsY) of the models. Unzip the folder, and move all of the files from the unzipped folder into the `models/` folder of your repo. Don't just drag the unzipped folder to the `models/` folder, you need to take out the individual files.

5. Run BeatMapSynth! The following line of code will output a custom Beat Saber map file in a zipped folder.

    ```python beatmapsynth.py ['filepath'] ['Song Name'] [difficulty] [model] [optional: -k] [optional: --version]```

    Where: 

    `['filepath']` is the path to your music file of choice

    `['Song Name']` is the name you wish to display in Beat Saber (will also be the name of the zip folder)

    `[difficulty]` is the desired difficulty level of the generated map, can be `'easy'`, `'normal'`, `'hard'`, `'expert'`, or `'expertPlus'`

    `[model]` is the model to use in mapping, can be `'random'`, `'HMM'`, `'segmented_HMM'`, or `'rate_modulated_segmented_HMM'`

    `[-k]` is the number of segments expected in the song for the segmented models. 5 is the default and seems to be a good fit for most songs following a standard verse-chorus structure. You may want to play around with this parameter if you find your mappings turn out too repetitive or not repetitive enough

    `[--version]` is the HMM model version. Can be `1` or `2`, 1 being a model trained with custom maps with ratings on beatsaver.com of over 90%, and 2 being trained on maps with rating over 70%. 2 is default.

    A full command might look like this in the Anaconda Prompt on Windows:

    ```python beatmapsynth.py "C:\Users\wvsha\Desktop\BeatMapSynthesizer\song.mp3" "Example_song" expert rate_modulated_segmented_HMM -k 4 --version 1```

    Or in a Mac terminal:

    ```python beatmapsynth.py '~/Desktop/music_files/song.mp3' 'Example Song' 'expert' 'rate_modulated_segmented_HMM' -k 4 --version 1```

    It generally takes less than 30 seconds to run. It will probably print a warning about PySoundFile not working and maybe something about a Numba deprecation warning, but this is fine. 
    
    After it finishes, a zip folder will appear in your working directory (i.e., the root folder of this repo). Unzip this folder, place it in the 'CustomMusic' folder in your Beat Saber system files, start Beat Saber, and find the map under Custom Music! 

To use BeatMapSynth again after initial installation, just navigate into your repo again, reactivate the BSMapSynth-env (`conda activate BSMapSynth-env`), then you'll be good to go.

That's it! Please enjoy and let me know how it works for you! If you encounter any bugs, feel free to submit an issue, or if you fix it yourself, submit a pull request! 