#!/usr/bin/env python

"""
Face Detection and Tracking System using Haar Cascades

This script implements real-time face detection and servo-based tracking
using OpenCV and Raspberry Pi GPIO.

Usage:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
"""

from __future__ import print_function
import sys
import getopt
import logging
from typing import List, Tuple

import numpy as np
import cv2 as cv
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

from video import create_capture
from common import clock, draw_str

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, servo_channels: int = 16, i2c_address: int = 0x40):
        """Initialize the face tracker with servo configuration."""
        self.servo_kit = ServoKit(channels=servo_channels, address=i2c_address)
        self.current_angle = 90.0  # Start at middle position
        self.prev_angle = 90.0
        self.min_angle = 0.0
        self.max_angle = 180.0

        # Constants for face detection
        self.SCALE_FACTOR = 1.3
        self.MIN_NEIGHBORS = 4
        self.MIN_FACE_SIZE = (30, 30)

        # Constants for servo movement
        self.MOVEMENT_THRESHOLD = 0.1  # 10% threshold for movement
        self.FRAME_WIDTH = 480  # Default frame width

    def detect_faces(self, img: np.ndarray, cascade: cv.CascadeClassifier) -> List[np.ndarray]:
        """Detect faces in the image using the provided cascade classifier."""
        try:
            rects = cascade.detectMultiScale(
                img,
                scaleFactor=self.SCALE_FACTOR,
                minNeighbors=self.MIN_NEIGHBORS,
                minSize=self.MIN_FACE_SIZE,
                flags=cv.CASCADE_SCALE_IMAGE
            )

            if len(rects) == 0:
                return []

            # Convert to x1,y1,x2,y2 format
            rects[:, 2:] += rects[:, :2]
            return rects

        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []

    def update_servo_position(self, face_x: int) -> None:
        """Update servo position based on face location."""
        try:
            # Calculate new angle based on face position
            new_angle = (face_x / self.FRAME_WIDTH) * self.max_angle
            new_angle = max(self.min_angle, min(new_angle, self.max_angle))

            # Check if movement is significant enough
            if (abs(new_angle - self.current_angle) / self.current_angle) > self.MOVEMENT_THRESHOLD:
                logger.debug(f"Moving servo to angle: {new_angle}")
                self.servo_kit.servo[0].angle = new_angle
                self.prev_angle = self.current_angle
                self.current_angle = new_angle

        except Exception as e:
            logger.error(f"Servo control error: {str(e)}")

    @staticmethod
    def draw_rects(img: np.ndarray, rects: List[np.ndarray], color: Tuple[int, int, int]) -> None:
        """Draw rectangles around detected faces."""
        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    """Main function to run the face detection and tracking system."""
    try:
        # Parse command line arguments
        args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
        video_src = video_src[0] if video_src else 0
        args = dict(args)

        # Initialize cascades
        cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
        nested_fn = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

        cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
        nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

        if cascade.empty() or nested.empty():
            raise ValueError("Error loading cascade classifiers")

        # Initialize camera
        cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(
            cv.samples.findFile('samples/data/lena.jpg')))

        # Initialize face tracker
        tracker = FaceTracker()

        while True:
            ret, img = cam.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            # Process frame
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.equalizeHist(gray)

            t = clock()

            # Detect and track faces
            rects = tracker.detect_faces(gray, cascade)
            vis = img.copy()
            tracker.draw_rects(vis, rects, (0, 255, 0))

            # Update servo for the first detected face
            if len(rects) > 0:
                x1, _, x2, _ = rects[0]
                face_center_x = (x1 + x2) // 2
                tracker.update_servo_position(face_center_x)

            # Detect eyes if nested cascade is available
            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = tracker.detect_faces(roi.copy(), nested)
                    tracker.draw_rects(vis_roi, subrects, (255, 0, 0))

            # Display performance metrics
            dt = clock() - t
            draw_str(vis, (20, 20), f'time: {dt*1000:.1f} ms')
            cv.imshow('facedetect', vis)

            if cv.waitKey(5) == 27:  # ESC key
                break

    except Exception as e:
        logger.error(f"Program error: {str(e)}")

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        GPIO.cleanup()
        cv.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()
