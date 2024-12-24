import threading
import time
import queue
from dataclasses import dataclass
from typing import List

import numpy as np

from camera.realsense_camera import RealsenseCamera
from robot_inteface.robot_interface import RobotInterface


@dataclass
class RobotState:
    timestamp: float
    joint_positions: List[float]
    tcp_pose: List[float]


class RobotRecorder:
    def __init__(self, controller: RobotInterface, sampling_rate=50):
        self.controller = controller
        self.sampling_rate = sampling_rate
        self.period = 1.0 / sampling_rate

        self.recording = False
        self.data_queue = queue.Queue()
        self.recording_thread = None

    def _record(self):
        while self.recording:
            cycle_start = time.time()

            state = RobotState(
                timestamp=cycle_start,
                joint_positions=self.controller.getActualQ(),
                tcp_pose=self.controller.getActualTCPPose(),
            )
            self.data_queue.put(state)

            processing_time = time.time() - cycle_start
            sleep_time = self.period - processing_time

            if sleep_time > 0:
                time.sleep(sleep_time)

    def start_recording(self):
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()

        recorded_data = []
        while not self.data_queue.empty():
            recorded_data.append(self.data_queue.get())
        return recorded_data

@dataclass
class RobotStateWithImage:
    timestamp: float
    joint_positions: List[float]
    tcp_pose: List[float]
    image_rgb: np.ndarray
    image_depth: np.ndarray


class RobotRecorderWithImage(RobotRecorder):
    def __init__(self, controller: RobotInterface, camera: RealsenseCamera, sampling_rate=50):
        super().__init__(controller, sampling_rate)
        self.camera = camera
        self.period = 1.0 / sampling_rate

    def _record(self):
        while self.recording:
            cycle_start = time.time()

            # Record data
            image_rgb, image_depth = self.camera.get_frame_rgb()
            state = RobotStateWithImage(
                timestamp=cycle_start,
                joint_positions=self.controller.getActualQ(),
                tcp_pose=self.controller.getActualTCPPose(),
                image_rgb=image_rgb,
                image_depth=image_depth,
            )
            self.data_queue.put(state)

            # Calculate remaining time and sleep if needed
            processing_time = time.time() - cycle_start
            sleep_time = self.period - processing_time

            if sleep_time > 0:
                time.sleep(sleep_time)