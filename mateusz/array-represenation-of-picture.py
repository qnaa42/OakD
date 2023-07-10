import cv2
import numpy as np
import depthai as dai


GridSize = 15

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

def GetMonoCamera(pipeline, isLeft):
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

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
        cv2.namedWindow("array representation")
        sideBySide = True

        while True:
            leftFrame = getFrame(leftQueue)
            rightFrame = getFrame(rightQueue)

            if sideBySide:
                frame = np.hstack((leftFrame, rightFrame))
            else:
                frame = np.uint8(leftFrame / 2 + rightFrame / 2)

            cv2.imshow("stereo pair", frame)

            #create copy of frame
            frameCopy = frame.copy()
            #split this copy in chunks of GridSizexGridSize pixels, then find the average value of each chunk, edit value of each pixel in the chunk to the average value, then add text in the middle of each chunk with the average value

            for i in range(0, frameCopy.shape[0], GridSize):
                for j in range(0, frameCopy.shape[1], GridSize):
                    chunk = frameCopy[i:i + GridSize, j:j + GridSize]
                    average = np.average(chunk)
                    frameCopy[i:i + GridSize, j:j + GridSize] = average
                    cv2.putText(frameCopy, str(int(average)), (j+int(GridSize/5), i+int(GridSize/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.15, (255, 255, 255), 1)


            #display this frame using cv2.imshow and name it "array representation"
            cv2.imshow("array representation", frameCopy)




            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                sideBySide = not sideBySide

        cv2.destroyAllWindows()