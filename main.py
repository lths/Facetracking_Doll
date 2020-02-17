#raspberry pi 3A
#install open_cv

#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import RPi.GPIO as GPIO



# local modules
from video import create_capture
from common import clock, draw_str

#duty cycle = 20ms/50h
#dc = x/20 * 100%

#initialize variables
servoPIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPIN,GPIO.OUT)
p = GPIO.PWM(servoPIN, 50)
p.start(0)
p.ChangeDutyCycle(0)
dcs = 0.0
old_dcs = 0.0



def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def move(dcs, old_dcs):
    
    if (0.9*old_dcs > dcs) or (dcs > old_dcs*1.1):
        print (dcs)
        print (old_dcs)
        p.ChangeDutyCycle(dcs)
        
    return dcs

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        
        #dc = int((x1+x2)/2)
        #print ("DC = ")
        #print(dc)
        #dcs = 50*(dc/480)
        #cv.rectangle(img, (dc, y1), (x2, y2), color, 2)
        #p.ChangeDutyCycle(dcs)
        
        
        

def main():
    import sys, getopt
    dcs = 0.0
    old_dcs = 0.0
    
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))

    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
            dc = int(((x1)+(x2))/2)
            dcs = 20*(dc/501)
            print ("DC = %f" % dcs)
            #print(dc)
            dcs = 20*(dc/501)
            
            #p.ChangeDutyCycle(dcs)
            old_dcs = move(dcs, old_dcs)
        
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

