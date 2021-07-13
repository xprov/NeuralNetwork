#!/usr/bin/env python3
import struct
from PIL import Image
from progress.bar import ChargingBar


def read32bitsBigEndian(binaryFile):
    return struct.unpack(">I", binaryFile.read(4))[0]

def read8bitsBigEndian(binaryFile):
    return int.from_bytes(struct.unpack(">c", binaryFile.read(1))[0], "big")

# Documentation from http://yann.lecun.com/exdb/mnist/
#
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
#
# The labels values are 0 to 9. 
#
def readLabels(filename):
    f = open(filename, 'rb')
    # first 4 bytes must be magic number 2049
    magicNumber = read32bitsBigEndian(f)
    if magicNumber != 2049:
        raise "Invalid file for labels, magic number (={}) is not 2049.".format(magicNumber)
    numEntries = read32bitsBigEndian(f)
    labels = []
    print("## Reading labels from " + filename)
    for i in range(numEntries):
        labels.append(read8bitsBigEndian(f))
    f.close()
    return labels
 
# Documentation from http://yann.lecun.com/exdb/mnist/
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
#
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background
# (white), 255 means foreground (black). 
#
def readImages(filename):
    f = open(filename, 'rb')
    # first 4 bytes must be magic number 2049
    magicNumber = read32bitsBigEndian(f)
    if magicNumber != 2051:
        raise "Invalid file for images, magic number (={}) is not 2051.".format(magicNumber)
    numEntries = read32bitsBigEndian(f)
    numRows = read32bitsBigEndian(f)
    numCols = read32bitsBigEndian(f)
    images = []
    bar = ChargingBar('## Reading images from ' + filename, max=numEntries, suffix='%(percent)d%%')
    for i in range(numEntries):
        im = Image.new("L", (numCols, numRows), 0)
        for i in range(numCols):
            for j in range(numRows):
                im.putpixel((j, i), 255-read8bitsBigEndian(f))
        images.append(im)
        bar.next()
    bar.finish()
    f.close()
    return images


def buildCSV(labelsFileName, imagesFileName, csvFileName):
    labels = readLabels(labelsFileName)
    images = readImages(imagesFileName)
    if len(labels) != len(images):
        raise "Invalid data, number of images does not match the number of labels"
    n = len(labels)
    f = open(csvFileName, 'w')
    bar = ChargingBar('## Writing to CSV file ', max=n, suffix='%(percent)d%%')
    for label, image in zip(labels, images):
        nbCols = image.height
        nbRows = image.width
        inputs = [1.0 - image.getpixel((x,y))/255.0 for y in range(nbRows) for x in range(nbCols)]
        outputs = [0]*10
        outputs[label] = 1
        f.write( ",".join(map(str, inputs))
               + ","
               + ",".join(map(str, outputs))
               + "\n")
        bar.next()
    bar.finish()
    f.close()

print("# Building training set...")
buildCSV("train-labels.idx1-ubyte", "train-images.idx3-ubyte", "training.csv")
print("# done")

print("# Building validation set...")
buildCSV("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "validation.csv")
print("# done")






