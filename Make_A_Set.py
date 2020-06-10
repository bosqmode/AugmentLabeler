import os
import random
import numpy as np
import cv2
import sys
import time
import shutil
import argparse

from scipy import ndimage
from DNetYoloV3Dataset import WriteYoloV3Set
from CropBbox import CropContents

# Commandline args
parser = argparse.ArgumentParser()
parser.add_argument("-setname", type=str, default='DATASET', help='The name of the folder and the dataset generated')
parser.add_argument("-huemin", type=float, default=1.0, help='Minimum range of the hue randomization')
parser.add_argument("-huemax", type=float, default=1.0, help='Maximum range of the hue randomization')
parser.add_argument("-tsize", type=int, default=1000, help='Size of the training set')
parser.add_argument("-vsize", type=int, default=100, help='Size of the validation set')
parser.add_argument("-rotate", default='True', help='Should the foregrounds be randomly rotated?')
parser.add_argument("-fgminfrac", type=float, default=0.4, help='Min size multiplier for a foreground')
parser.add_argument("-fgmaxfrac", type=float, default=1.2, help='Max size multiplier for a foreground')
parser.add_argument("-alphacrop", default='True', help="Should the foregrounds be cropped by their alpha channel's bounding box?")
parser.add_argument("-includenegative", default='False', help='For every image/label of a foreground, add one "negative"-image. Basically just copies a background image to the training/validation sets. (NOTE: This will double the size of training/validation data)')
args = parser.parse_args()

# Directories
fg_directory = "foregrounds"
bg_directory = "backgrounds"
train_images = os.path.join(args.setname, "images", "Train")
val_images = os.path.join(args.setname, "images", "Val")
train_labels = os.path.join(args.setname, "labels", "Train")
val_labels = os.path.join(args.setname, "labels", "Val")

args.rotate = args.rotate.lower() in ['true', '1']
args.includenegative = args.includenegative.lower() in ['true', '1']
args.alphacrop = args.alphacrop.lower() in ['true', '1']

bg_images = []

# Blends the foreground to the background starting at startX and startY, and ending at endX and endY
def AlphaBlend(foreground, background, startX, endX, startY, endY):
    alpha_s = foreground[0:endY, 0:endX, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        background[startY:endY, startX:endX, c] = (alpha_s * foreground[0:endY, 0:endX, c] + alpha_l * background[startY:endY, startX:endX, c])

    return background
    

# def Overlay(foreground, background, startX, endX, startY, endY):
#     for c in range(0, 3):
#         background[startY:endY, startX:endX, c] = (foreground[:, :, c])

#     #cv2.imshow("img overlay", background/255)
#     #cv2.waitKey(0)

#     return background

# Clears a directory, and it's subfolders, of files
def ClearDir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Makes, and/or clears, the folders for the datasets
def MakeAndClearDirs():
    if not os.path.exists(args.setname):
        os.mkdir(args.setname)

    if not os.path.exists(os.path.join(args.setname, "images")):
        os.mkdir(os.path.join(args.setname, "images"))

    if not os.path.exists(os.path.join(args.setname, "labels")):
        os.mkdir(os.path.join(args.setname, "labels"))

    if not os.path.exists(train_images):
        os.mkdir(train_images)
    else:
        ClearDir(train_images)
    
    if not os.path.exists(val_images):
        os.mkdir(val_images)
    else:
        ClearDir(val_images)

    if not os.path.exists(train_labels):
        os.mkdir(train_labels)
    else:
        ClearDir(train_labels)

    if not os.path.exists(val_labels): 
        os.mkdir(val_labels)
    else:
        ClearDir(val_labels)

# Randomizes the hue/colors of an image
def RandomizeHue(image):
    (r, g, b) = cv2.split(image)
    r = r*random.uniform(args.huemin, args.huemax)
    g = g*random.uniform(args.huemin, args.huemax)
    b = b*random.uniform(args.huemin, args.huemax)
    return cv2.merge([r,g,b])

# Gets the current timestamp as a string
def UnixTS():
    return str( int(round(time.time() * 1000)) )

# Adds a negative image(negative in this case; it has none of the classes we are trying to train on) (basically just one of the background images copied)
def AddNegative(filename, targetdir):
    shutil.copy2(os.path.join(bg_directory, filename), os.path.join(targetdir, UnixTS() + ".jpg"))

# Process of merging a foreground and a background image
def Process(fg, bg, train, classindex):

    # Read the images
    background = cv2.imread(bg)
    foreground = cv2.imread(fg, -1)

    # add the Alpha channel to .jpegs for future processing
    if not fg.lower().endswith(".png"):
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
        foreground[...,3] = 255

    # Resize the foreground by fgminfrac fgmaxfrac
    randomscale = random.uniform(args.fgminfrac, args.fgmaxfrac)
    width = int(foreground.shape[1] * randomscale)
    height = int(foreground.shape[0] * randomscale)
    dim = (width, height)
    foreground = cv2.resize(foreground, (dim))

    # Rotate the foreground
    if args.rotate == True:
        foreground = ndimage.rotate(foreground, 360 * random.random())

    # Try to keep the background bigger than the foreground, so the foreground does not clip
    # since this might add odd/false entries to the dataset.
    if background.shape[0] < foreground.shape[0] or background.shape[1] < foreground.shape[1]:
        background = cv2.resize(background, ((int)(foreground.shape[1]+foreground.shape[1]*0.1), (int)(foreground.shape[0]+foreground.shape[0]*0.1)))

    # Offset by random amount
    offsetY = int((background.shape[0]-foreground.shape[0]) * random.random())
    offsetX = int((background.shape[1]-foreground.shape[1]) * random.random())

    y1, y2 = offsetY, offsetY + foreground.shape[0]
    x1, x2 = offsetX, offsetX + foreground.shape[1]

    y1 = np.clip(y1, 0, background.shape[0])
    y2 = np.clip(y2, 0, background.shape[0])
    x1 = np.clip(x1, 0, background.shape[1])
    x2 = np.clip(x2, 0, background.shape[1])

    # Blend the foreground to the background with the positions calculated above
    image = AlphaBlend(foreground, background, x1, x2, y1, y2)

    # Change the hue of the image if required
    if args.huemin is not 1.0 or args.huemax is not 1.0:
        image = RandomizeHue(image)

    # Save to the training folder's and write the labels
    name = UnixTS()

    if train == True:
        Write(name, image, classindex, y1, y2, x1, x2, train_images, train_labels, background, foreground)
    else:
        Write(name, image, classindex, y1, y2, x1, x2, val_images, val_labels, background, foreground)

# Writes the processed image to the dataset and writes a label for that image
def Write(name, image, classIndex, y1, y2, x1, x2, imagefolder, labelfolder, bg, fg):
    cv2.imwrite(os.path.join(imagefolder, name + ".jpg"),image)
    f = open(os.path.join(labelfolder, name + ".txt"),"w+")
    # class x_center y_center width height
    f.write(str(classIndex) + " " + str((x1 + (x2-x1)/2)/bg.shape[1]) + " " + str((y1 + (y2-y1)/2)/bg.shape[0]) + " " + str(fg.shape[1]/bg.shape[1]) + " " + str(fg.shape[0]/bg.shape[0]))

def PrintProgress(processedcount):
    print(str(round(processedcount / (args.tsize + args.vsize)*100, 2)) + "%...", end="\r")
    time.sleep(0.0001)

# The loop of making the dataset
def Bake(tsize, vsize):
    classidx = 0

    processed = 0

    # loop through all of the folders in foregrounds-directory, these folders will be the different classes being labeling
    for dirname in os.listdir(fg_directory):
        dirpath = os.path.join(fg_directory, dirname)
        if os.path.isdir(dirpath):

            # Get the different images of the class in the folder and loop through those one by one so the dataset should have equal amount of every single image
            filesInFolder = os.listdir(dirpath)

            fgidx = 0
            for x in range(tsize):
                fgidx += 1
                if fgidx >= len(filesInFolder):
                    fgidx = 0
                if filesInFolder[fgidx].lower().endswith(".jpg") or filesInFolder[fgidx].lower().endswith(".png"):
                    Process(os.path.join(dirpath, filesInFolder[fgidx]), os.path.join(bg_directory, random.choice(bg_images)), True, classidx)
                    if args.includenegative == True:
                        AddNegative(random.choice(bg_images), train_images)

                processed += 1
                PrintProgress(processed)

            for y in range(vsize):
                fgidx += 1
                if fgidx >= len(filesInFolder):
                    fgidx = 0
                if filesInFolder[fgidx].lower().endswith(".jpg") or filesInFolder[fgidx].lower().endswith(".png"):
                    Process(os.path.join(dirpath, filesInFolder[fgidx]), os.path.join(bg_directory, random.choice(bg_images)), False, classidx)
                    if args.includenegative == True:
                        AddNegative(random.choice(bg_images), val_images)

                processed += 1
                PrintProgress(processed)

        classidx += 1


if __name__ == "__main__":

    #Preprocess
    MakeAndClearDirs()
    if args.alphacrop == True:
        CropContents(fg_directory)
    for filename in os.listdir(bg_directory):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            bg_images.append(filename)
    classes = len(os.listdir(fg_directory))
    train_per_class = (int)(args.tsize / classes)
    val_per_class = (int)(args.vsize / classes)

    #Bake the set!
    Bake(train_per_class, val_per_class)
    WriteYoloV3Set(args.setname, train_images, val_images, os.listdir(fg_directory))
    print("Done!")
    