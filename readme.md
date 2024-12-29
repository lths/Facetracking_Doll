# Face Detection and Tracking System

A Python-based face detection and tracking system using OpenCV's Haar Cascades, designed to run on a Raspberry Pi with servo motor control. The system detects faces in real-time video feed and adjusts a servo motor to track the detected face.

## Features

- Real-time face detection using Haar Cascades
- Optional eye detection (nested cascade)
- Servo motor control for face tracking
- Support for both video files and live camera feed
- Compatible with both Python 2 and 3

## Hardware Requirements

- Raspberry Pi (any model with GPIO pins)
- Servo Motor
- Camera (USB webcam or Raspberry Pi Camera Module)
- I2C-capable servo controller (Adafruit 16-channel PWM/Servo HAT)

## Software Dependencies

```bash
pip install numpy opencv-python RPi.GPIO adafruit-circuitpython-servokit
