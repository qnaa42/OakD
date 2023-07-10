import numpy as np
import cv2
import depthai as dai
import blobconverter

pipeline = dai.Pipeline()

# Define a source - color camera (for each of the 3 streams separately)
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)

detection_nn = pipeline.createMobileNetDetectionNetwork()

# Set path to blob
detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
detection_nn.setConfidenceThreshold(0.5)

# connect camera to neural network
camRgb.preview.link(detection_nn.input)


# parsing output from neural network and camera frame to host using XLink connection

# Camera frame output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
camRgb.preview.link(xout_rgb.input)

# Neural network output
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# connect to device and start pipeline
with dai.Device(pipeline) as device:
    # grabbing output from camera frame and neural network
    q_rgb = device.getOutputQueue(name="rgb")
    q_nn = device.getOutputQueue(name="nn")

    # placeholders for frame and detection objects
    frame = None
    detections = []

    # bounding boxes coordinates are normalized so need to multiply with frame width and height
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections

        #if frame exist copy it
        if frame is not None:
            frameCopy = frame.copy()

        if frame is not None and len(detections) > 0:
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("rgb", frame)

        if frame is not None and len(detections) > 0:
        # split this copy in chunks of GridSizexGridSize pixels, then find the average value of each chunk, edit value of each pixel in the chunk to the average value, then add text in the middle of each chunk with the average value
            GridSize = 10 * len(detections)
            for i in range(0, frameCopy.shape[0], GridSize):
                for j in range(0, frameCopy.shape[1], GridSize):
                    # find the average value of each chunk
                    chunk = frameCopy[i:i+GridSize, j:j+GridSize]
                    average = np.mean(chunk)
                    # edit value of each pixel in the chunk to the average value
                    frameCopy[i:i+GridSize, j:j+GridSize] = average
                    # add text in the middle of each chunk with the average value
                    cv2.putText(frameCopy, str(round(average, 2)), (j+int(GridSize/2), i+int(GridSize/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("rgbCopy", frameCopy)


        if cv2.waitKey(1) == ord('q'):
            break


