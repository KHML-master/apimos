#TODO Update this read me
# Data Preprocessing
Two data preprocessing phases are implemented to ease the process of manipulating data.

### Table of Contents
- [Data Preprocessing](#data-preprocessing)
    - [Table of Contents](#table-of-contents)
  - [Video to Frames](#video-to-frames)
    - [How to Run](#how-to-run)
  - [Sort Annotated Images](#sort-annotated-images)
    - [How to Run](#how-to-run-1)


---

## Video to Frames
This phase takes a directory of videos and converts them into images. Moreover, it removes both dark images and grayscaled images which are due to night and inferred respectively.

### How to Run
Enter the directory `/data_preprocessing/video_to_frames/` and copy all the videos into `./input_data/`, thereafter run:
```
python video_to_frames_class.py
```
This will output images into the directory `./output_data/`.

Outputting discarded images can be done by adding `--debug_images true` to the previous command.

---

## Sort Annotated Images
When exporting a dataset from CVAT, the data directory is a single directory consisting of both labeled and not labeled data. This phase seperates the data into two directories depending on the labeling status. Moreover, it writes two txt files `train.txt` and `test.txt` that includes the names of the images, where names of labeled images are put into `train.txt` and non-labeled images into `test.txt`.

### How to Run
Enter the directory `/data_preprocessing/sort_annotated_images/` and copy all the videos into `./input_data/`, thereafter run:
```
python delete_images_with_no_bbx.py
```
This will output the dataset into the directory `./output_data/`.