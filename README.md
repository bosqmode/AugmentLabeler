# AugmentLabeler

A tool for making a training dataset for image classification, in
a situation where one has just a few images of the object to recognize. (targeted mainly for Darknet/YoloV3) 

![Intro](https://i.imgur.com/MV8S3F4.jpg)

### Requirements (tested to work with)

Python 3.7.5

numpy 1.17.3
scipy 1.4.1
Pillow 6.2.1
opencv-python 4.1.0
argparse 1.4.0

``` pip3 install -U -r requirements.txt ```

## Usage

Install the requirements mentioned above.

Create a directory for every class in the "foregrounds"-directory (the name of the created directory will be used as the name of this class).
Fill the foreground-folders with images of the classes.

Fill the "backgrounds"-folder with images that do not contain ANY of the classes being recognized.

Run ``` python Make_A_Set.py ``` (Note arguments explained below).

If all goes well, one should see a new folder: DATASET,
in which resides all of the labels and images.

``` images ``` -folder contains all of the augmented images themselves.
and ``` labels ``` -folder contains label txt-files for
those images. These are in Darknet format and one can find more of that here: https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
``` .names ``` -file has all of the classnames written into it.

and for Darknet/Yolo -users the script will also generate the darknet-format train and validation .txt -files

Also a Darknet/Yolo -type ``` .data ``` -file will be generated next to the DATASET-folder itself. Since all of the path's are written in absolute path, training for YoloV3 should be possible just by passing the YoloV3's train.py with ``` --data {{pathto/DATASET.data}} ```


## Make_A_Set.py arguments with default values

There is a set of settings (no pun intended) to tailor the dataset for the user's needs
Example usage: ``` python Make_A_Set.py -tsize=20 -vsize=10 -rotate=False -alphacrop=False ```

### -setname
``` -setname=DATASET ```
The name of the folder and the dataset generated

### -huemin -huemax
``` -huemin=1.0 -huemax=1.0 ```
Minimum and maximum ranges of the hue randomization
Not on by default since most of the DNN-frameworks are already
equipped with such functionality.

![Hue1](https://i.imgur.com/bqgTYl1.png)

``` -huemin=0.6 -huemax=1.5 ```

![Hue2](https://i.imgur.com/dDh3pIr.png)

### -tsize -vsize
``` -tsize=1000 -vsize=100 ```
Sizes of the training and validation sets respectively

### -rotate
``` -rotate=True ```
Randomly rotates the foregrounds

### -fgminfrac -fgmaxfrac
``` -fgminfrac=0.4 -fgmaxfrac=1.2 ```
Size multiplier range for foregrounds

### -alphacrop
``` -alphacrop=True ```
Should the foregrounds be cropped by their alpha channel's bounding box? Note: this is done in
preprocessing and therefor replaces the old images
with cropped ones. To be safe, make a copies of the original ones.

![Cropped](https://i.imgur.com/bvtDFo3.jpg)

``` -alphacrop=False ```

![UnCropped](https://i.imgur.com/oH18bFq.jpg)

### -includenegative
``` -includenegative=False ```
For every image/label of a foreground, add one "negative"-image. Basically just copies a background image to the training/validation sets without augmenting a foreground onto it. (NOTE: This will double the size of training/validation data)

## Acknowledgments

* Inspiration: https://memememememememe.me/post/training-haar-cascades/

* More on YoloV3 usage and custom training: https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data

