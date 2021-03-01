import os
import numpy as np
import stitch
import cv2
from pyspark import SparkContext

def stitch_spark(point):
    frame, bs = point

    # Get numpy arrays for images
    arrs = [np.frombuffer(b, np.uint8) for b in bs]

    # Decode all images
    imgs = [cv2.imdecode(arr, 1) for arr in arrs]

    # Stitch
    simg = stitch.ImageStitcher().blending(imgs[0], imgs[1])
    simgb = cv2.imencode('.jpg', simg)[1]

    return frame, simgb

def get_frame(point):
    filename, data = point

    # process filename (frame specifier detail)
    base = os.path.basename(filename)  # remove path
    frame_name = os.path.splitext(base)[0]  # remove extension

    return (frame_name, data)

if __name__ == "__main__":
    sc = SparkContext(appName="stitcher")
    sc = SparkContext.getOrCreate()

    # Read all input
    source1 = sc.binaryFiles("file:///home/vostro/cs219/images1")  # first source
    source2 = sc.binaryFiles("file:///home/vostro/cs219/images2")  # second source

    # Get frame-data mapping for all sources
    source1 = source1.map(get_frame)
    source2 = source2.map(get_frame)

    # Join all sources into single RDD
    sources = source1.join(source2)

    # Stitch!
    stitched = sources.map(stitch_spark)

    # Get results
    stitched_frames = stitched.collect()

    # Output
    output_dir = '/home/vostro/cs219/output/'
    for frame in stitched_frames:
        frame_name, frame_bytes = frame
        with open(output_dir + frame_name + '.jpg', 'wb') as f:
            frame_bytes.tofile(f)
