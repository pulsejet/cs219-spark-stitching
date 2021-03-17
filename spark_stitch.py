import os
import numpy as np
import stitch
import cv2
import glob
import re
from pyspark import SparkContext

def stitch_spark(point):
    frame, bs = point

    # Get numpy arrays for images
    # arrs = [np.frombuffer(b, np.uint8) for b in bs]

    # Decode all images
    # imgs = [cv2.imdecode(arr, 1) for arr in arrs]
    imgs = bs

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

    # Read all from files
    # source1 = sc.binaryFiles("file:///home/vostro/cs219/images1")  # first source
    # source2 = sc.binaryFiles("file:///home/vostro/cs219/images2")  # second source
    # sources = [source1, source2]  # array of RDDs of all sources

    source1 = cv2.VideoCapture("/home/vostro/cs219/left.mp4")
    source2 = cv2.VideoCapture("/home/vostro/cs219/right.mp4")

    success1 = 1
    success2 = 1
    count1 = 1
    count2 = 1

    while success1 and success2:
        s1l = []
        s2l = []

        for i in range(0, 50):
            success1, image1 = source1.read()
            success2, image2 = source2.read()
            if not success1 or not success2:
                break

            s1l.append((count1, image1))
            s2l.append((count2, image2))

            count1 += 1
            count2 += 1

        sources = [sc.parallelize(s1l), sc.parallelize(s2l)]

        # Get frame-data mapping for all sources
        # sources = [s.map(get_frame) for s in sources]

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
            with open(output_dir + str(frame_name).zfill(5) + '.jpg', 'wb') as f:
                frame_bytes.tofile(f)

    # Create video
    img_array = []
    filnames = sorted([os.path.basename(x) for x in glob.glob('/home/vostro/cs219/output/*.jpg')])

    size1 = None

    for filename in filnames:
        img = cv2.imread('/home/vostro/cs219/output/' + filename)
        height, width, layers = img.shape
        size = (width,height)
        if size1:
            resized = cv2.resize(img,size1)
            img_array.append(resized)
        else:
            size1 = size
            img_array.append(img)

        print(filename)

    out = cv2.VideoWriter('/home/vostro/cs219/out.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size1)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
