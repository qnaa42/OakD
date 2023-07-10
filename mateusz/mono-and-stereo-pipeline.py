import cv2
import numpy as np
import depthai as dai

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

def GetMonoCamera(pipeline, isLeft):
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    return mono

if __name__ == '__main__':
    pipeline = dai.Pipeline()
    left = GetMonoCamera(pipeline, True)
    right = GetMonoCamera(pipeline, False)

    xOutLeft = pipeline.createXLinkOut()
    xOutLeft.setStreamName("left")

    xOutRight = pipeline.createXLinkOut()
    xOutRight.setStreamName("right")

    left.out.link(xOutLeft.input)
    right.out.link(xOutRight.input)

    with dai.Device(pipeline) as device:
        leftQueue = device.getOutputQueue(name="left", maxSize=1)
        rightQueue = device.getOutputQueue(name="right", maxSize=1)

        cv2.namedWindow("stereo pair")
        sideBySide = True

        while True:
            leftFrame = getFrame(leftQueue)
            rightFrame = getFrame(rightQueue)

            if sideBySide:
                frame = np.hstack((leftFrame, rightFrame))
            else:
                frame = np.uint8(leftFrame / 2 + rightFrame / 2)

            cv2.imshow("stereo pair", frame)
            #display array representation of image in the console
            print(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                sideBySide = not sideBySide