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
    # TODO: support stitching >2 sources
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
    sources = [source1, source2]  # array of RDDs of all sources

    # Get frame-data mapping for all sources
    sources = [s.map(get_frame) for s in sources]

    # Join all sources into single RDD
    # TODO: find a more generic way for this that works for >2
    sources = sources[0].join(sources[1])

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
