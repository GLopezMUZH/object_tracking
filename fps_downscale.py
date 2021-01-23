import cv2


PATH_TO_VIDEO = '/Users/pascal/Coding/MP_bees/simple_object_tracking/videos/118_Doettingen_Hive1_200820_gopro8_1080_200fps_W_short.mp4'

cap = cv2.VideoCapture(PATH_TO_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



outname = '118_Doettingen_Hive1_200820_gopro8_1080_200fps_W_short_100fps.avi'
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(outname, fourcc, 10.0, (int(width), int(height)))

def getFrame(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = cap.read()
    if hasFrames:
        out.write(cv2.resize(image, (int(width), int(height))))
    return hasFrames
sec = 0
frameRate = 1
success = getFrame(sec)
while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    sec += 1

cap = cv2.VideoCapture(PATH_TO_VIDEO)
cap.set(cv2.CAP_PROP_FPS, 150)
print(int(cap.get(cv2.CAP_PROP_FPS)))
