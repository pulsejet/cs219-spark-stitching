import cv2

def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1

    while success:
        success, image = vidObj.read()
        try:
            cv2.imwrite("vid/frame%d.jpg" % count, image)
            count += 1
        except Exception:
            print('Skip at', count)

if __name__ == '__main__':
    FrameCapture("left.mp4")