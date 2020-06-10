import os
import pathlib

def WriteDataSet(target_filename, imagefolder, abspath):
    f = open(target_filename,"w+")
    for filename in os.listdir(imagefolder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            f.write(os.path.join(abspath, imagefolder, filename) +"\n")

def WriteNamesFile(classes, filename):
    f = open(filename,"w+")
    for name in classes:
        f.write(name + "\n")

def WriteDataFile(datasetname, classes, tsetname, vsetname, namesname):
    f = open(datasetname + ".data","w+")
    f.write("classes=" + str(classes) + "\n")
    f.write("train=" + tsetname + "\n")
    f.write("valid=" + vsetname + "\n")
    f.write("names=" + namesname)

def WriteYoloV3Set(datasetname, train_images, val_images, classes):
    abspath = pathlib.Path(__file__).parent.absolute()

    trainset = os.path.join(abspath, datasetname, datasetname + "_train.txt")
    valset = os.path.join(abspath, datasetname, datasetname + "_val.txt")
    names = os.path.join(abspath, datasetname, datasetname + ".names")

    WriteDataSet(trainset, train_images, abspath)
    WriteDataSet(valset, val_images, abspath)
    WriteNamesFile(classes, names)
    WriteDataFile(datasetname, len(classes), trainset, valset, names)
