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

def get_dots_bounds(dots):
    """Returns the bounds of a list of dots."""

    return (
        min(x for x, y in dots),
        max(x for x, y in dots),
        min(y for x, y in dots),
        max(y for x, y in dots)
    )

def rescale_dots(a, b):
    """Scales, offsets and returns a to match the bounds of b."""

    a_left, a_right, a_top, a_bottom = get_dots_bounds(a)
    b_left, b_right, b_top, b_bottom = get_dots_bounds(b)
    scale_x = (b_right - b_left) / (a_right - a_left)
    scale_y = (b_bottom - b_top) / (a_bottom - a_top)
    return [
        ((x - a_left) * scale_x + b_left, (y - a_top) * scale_y + b_top)
        for x, y
        in a
    ]

def generate_pairs(reference, scanned):
    """Generate a list of corresponding dots from two images."""

    # This might break if the scan is so distorted that
    # the rows or columns begin to overlap
    reference_dots = get_dots(reference)
    logging.debug('Found %d reference dots' % len(reference_dots))

    scanned_dots = get_dots(scanned)
    logging.debug('Found %d scanned dots' % len(scanned_dots))
    assert len(reference_dots) == len(scanned_dots), ERR_DOT_COUNT

    rescaled_reference_dots = rescale_dots(reference_dots, scanned_dots)

    pairs = []
    for rdot in rescaled_reference_dots:
        sdot = min(scanned_dots, key=lambda sdot: distance(rdot, sdot))
        pairs.append((rdot, sdot))

    return pairs

def scale_pairs(pairs, scale):
    """"Scale an array of pairs."""

    return [
        ((rx * scale[0], ry * scale[1]), (sx * scale[0], sy * scale[1]))
        for (rx, ry), (sx, sy)
        in pairs
    ]

def undistort(image, pairs):
    """Undistort an image."""

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
        scale = (
            image.size[0] / WORKING_SIZE[0],
            image.size[1] / WORKING_SIZE[1]
        )
        scaled_pairs = scale_pairs(pairs, scale)
        undistort(image, scaled_pairs)
        image.save(filename=output[i])
