# bzimage
*Various example output of a worst case scenario: uneven lighting across each field*
![Example Output](example%20output%20github/example.jpg)

full focuses (MIP) and merges (fast no-correction concatenation) 4-color images from the Keyence BZ-700 series microscope

[example data (~300MB)](https://hu-my.sharepoint.com/:u:/g/personal/wesley_wong_fas_harvard_edu/EXhaOkXjy45ItkoJyrVsXbAB_x7vQusD6TKXUH4tSYJ-OA?e=d2sa1t)


[stand alone bundles for MacOS and Windows 10](https://hu-my.sharepoint.com/:f:/g/personal/wesley_wong_fas_harvard_edu/EsEbDkv3j9RFjH1KcD27DrsBCfJR8t2FHYGr-AMQw2-Njg?e=hmxVCf)


written under python 3.7.7 using anaconda- see requires.txt for packages needed.

how to use:

python bzimage3.py "target folder path"
  

the main advantage of this script is that for a typical tiled and z-stacked image capture, it's normal for ~5000+ .tif images to be produced (0.5-1mb each) and the official software for handling full focus and merging takes about 5 minutes and constant clicking through menus to achieve.


this script generates a fast full focus merged image in about 10-30 seconds and also outputs full focused individual tile images that can be stitched in photoshop or FIJI (see example output) to achieve similar results.

this script supports maximum intensity projections (default) or maximum contrast projections (using "--contrast"), adopted from an R package authored by user DragonDuck here: https://github.com/DragonDuck/MaxContrastProjection


1-24-21 TODO:

--I have a working command line + automator app Mac bundle of the script made with pyinstaller, but building the bundle for Windows produces a .exe that only works when run from the anaconda prompt, not the regular command line. It would be nice to get this fixed and implement a batch script to simplify the usage.


-expose runtime options like manual grid size and tile overlap amount


-fix halos around bright objects in maximum contrast mode that occur after guassian blur


-corrective planar transformations and keypoint/RANSAC based tile merging (as a run option, like --highquality)

