import depthai as dai
import blobconverter
import cv2
import numpy as np
from pythonosc import udp_client



label_map = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Setup OSC client
osc_client = udp_client.SimpleUDPClient("localhost", 12345)

#prediction class
class KalmanFilter(object):
    def __init__(self, acc_std, meas_std, z, time):
        self.dim_z = len(z)
        self.time = time
        self.acc_std = acc_std
        self.meas_std = meas_std

        # the observation matrix
        self.H = np.eye(self.dim_z, 3 * self.dim_z)

        self.x = np.vstack((z, np.zeros((2 * self.dim_z, 1))))
        self.P = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        i, j = np.indices((3 * self.dim_z, 3 * self.dim_z))
        self.P[(i - j) % self.dim_z == 0] = 1e5  # initial vector is a guess -> high estimate uncertainty

    def predict(self, dt):
        # the state transition matrix -> assuming acceleration is constant
        F = np.eye(3 * self.dim_z)
        np.fill_diagonal(F[:2 * self.dim_z, self.dim_z:], dt)
        np.fill_diagonal(F[:self.dim_z, 2 * self.dim_z:], dt ** 2 / 2)

        # the process noise matrix
        A = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        np.fill_diagonal(A[2 * self.dim_z:, 2 * self.dim_z:], 1)
        Q = self.acc_std ** 2 * F @ A @ F.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        if z is None: return

        # the measurement uncertainty
        R = self.meas_std ** 2 * np.eye(self.dim_z)

        # the Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(3 * self.dim_z)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T


# Create pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
detection_network = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
object_tracker = pipeline.create(dai.node.ObjectTracker)

xout_rgb = pipeline.create(dai.node.XLinkOut)
tracker_out = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName('rgb')
tracker_out.setStreamName('tracklets')

cam_rgb.setPreviewSize(300, 300)
cam_rgb.setVideoSize(1920, 1080)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

detection_network.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=5))
detection_network.setConfidenceThreshold(0.7)
detection_network.input.setBlocking(False)
detection_network.setBoundingBoxScaleFactor(0.5)
detection_network.setDepthLowerThreshold(100)
detection_network.setDepthUpperThreshold(5000)

object_tracker.setDetectionLabelsToTrack([15])  # track only person
object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

cam_rgb.preview.link(detection_network.input)
object_tracker.passthroughTrackerFrame.link(xout_rgb.input)
object_tracker.out.link(tracker_out.input)

detection_network.passthrough.link(object_tracker.inputTrackerFrame)
detection_network.passthrough.link(object_tracker.inputDetectionFrame)
detection_network.out.link(object_tracker.inputDetections)
stereo.depth.link(detection_network.inputDepth)

with dai.Device(pipeline) as device:
    calibration_handler = device.readCalibration()
    baseline = calibration_handler.getBaselineDistance() * 10
    focal_length = calibration_handler.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 640, 400)[0][0]

    q_rgb  = device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
    q_tracklets = device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)

    kalman_filters = {}

    while(True):
        frame = q_rgb.get().getCvFrame()
        frame = cv2.resize(frame, (1920, 1080))  # Resize the frame to Full HD
        tracklets = q_tracklets.get()
        current_time = tracklets.getTimestamp()

        h, w = frame.shape[:2]
        for t in tracklets.tracklets:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            x_space = t.spatialCoordinates.x
            y_space = t.spatialCoordinates.y
            z_space = t.spatialCoordinates.z

            meas_vec_bbox = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]])
            meas_vec_space = np.array([[x_space], [y_space], [z_space]])
            meas_std_space =  z_space ** 2 / (baseline * focal_length)

            if t.status.name == 'NEW':
                # Adjust these parameters
                acc_std_space = 10
                acc_std_bbox = 0.1
                meas_std_bbox = 0.05

                kalman_filters[t.id] = {'bbox': KalmanFilter(meas_std_bbox, acc_std_bbox, meas_vec_bbox, current_time),
                                        'space': KalmanFilter(meas_std_space, acc_std_space, meas_vec_space, current_time)}
            else:
                dt = current_time - kalman_filters[t.id]['bbox'].time
                dt = dt.total_seconds()
                kalman_filters[t.id]['space'].meas_std = meas_std_space

                if t.status.name != 'TRACKED':
                    meas_vec_bbox = None
                    meas_vec_space = None

                if z_space == 0:
                    meas_vec_space = None

                kalman_filters[t.id]['bbox'].predict(dt)
                kalman_filters[t.id]['bbox'].update(meas_vec_bbox)

                kalman_filters[t.id]['space'].predict(dt)
                kalman_filters[t.id]['space'].update(meas_vec_space)

                kalman_filters[t.id]['bbox'].time = current_time
                kalman_filters[t.id]['space'].time = current_time

                vec_space = kalman_filters[t.id]['space'].x

                # Broadcast detection coordinates using OSC
                osc_client.send_message("/detection", [int(vec_space[0]), int(vec_space[1]), int(vec_space[2])])

                # Display the coordinates in the top left corner
                cv2.putText(frame, f'X: {int(vec_space[0])} mm', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255))
                cv2.putText(frame, f'Y: {int(vec_space[1])} mm', (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255))
                cv2.putText(frame, f'Z: {int(vec_space[2])} mm', (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255))

        cv2.imshow('tracker', frame)
        if cv2.waitKey(1) == ord('q'):
            break
