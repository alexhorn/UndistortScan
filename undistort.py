import argparse
import cv2
import logging
import numpy as np
import os
from wand.image import Image

ERR_FILE_READ =\
'Could not read %s'
ERR_DOT_COUNT =\
'Number of dots in reference and scanned is different (your scan might be bad)'
IMG_REFERENCE='reference.png'
IMG_SCANNED='scanned.png'
LOG_LEVEL='debug'

BLOB_MIN_THRESHOLD=127
BLOB_MAX_THRESHOLD=255
BLOB_MIN_AREA=64
BLOB_MAX_AREA=255
WORKING_SIZE=(1700, 2338) # DIN A4 page at 200 dpi

def distance(a, b):
    """Calculate the distance between two points a and b."""

    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def get_dots(filename):
    """Find the dots in an image."""

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    assert img is not None, ERR_FILE_READ % filename

    img = cv2.resize(img, WORKING_SIZE, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = BLOB_MIN_THRESHOLD
    params.maxThreshold = BLOB_MAX_THRESHOLD
    params.filterByArea = True
    params.minArea = BLOB_MIN_AREA
    params.maxArea = BLOB_MAX_AREA
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    return [kp.pt for kp in keypoints]

def generate_pairs(reference, scanned):
    """Generate a list of corresponding dots from two images."""

    # This might break if the scan is so distorted that
    # the rows or columns begin to overlap
    reference = get_dots(reference)
    logging.debug('Found %d reference dots' % len(reference))

    scanned = get_dots(scanned)
    logging.debug('Found %d scanned dots' % len(scanned))

    assert len(reference) == len(scanned), ERR_DOT_COUNT

    pairs = []
    for rdot in reference:
        sdot = min(scanned, key=lambda x:distance(rdot, x))
        pairs.append((rdot, sdot))

    return pairs

def undistort(image, pairs):
    """Undistort an image"""
    arguments = []
    for rdot, sdot in pairs:
        arguments.extend(sdot)
        arguments.extend(rdot)

    image.distort('perspective', arguments)

    return image

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Undistorts a scanned image.')
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--reference', type=str, default=IMG_REFERENCE)
parser.add_argument('--scanned', type=str, default=IMG_SCANNED)
args = parser.parse_args()

if os.path.isdir(args.input):
    if not os.path.exists(args.output):
        os.path.mkdir(args.output)

    assert os.path.isdir(args.output)

    files = os.listdir(args.input)
    input = list(map(lambda file: os.path.join(args.input, file), files))
    output = list(map(lambda file: os.path.join(args.output, file), files))
else:
    input = [args.input]
    output = [args.output]

logging.info('Generating undistortion map')
pairs = generate_pairs(args.reference, args.scanned)

for i in range(len(input)):
    logging.info('Processing %s' % input[i])

    with Image(filename=input[i]) as image:
        undistort(image, pairs)
        image.save(filename=output[i])
