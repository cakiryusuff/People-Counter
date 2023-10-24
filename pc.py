# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:34:55 2023

@author: cakir
"""

import cv2

cam = cv2.VideoCapture("TestVideo.avi")

backSub = cv2.createBackgroundSubtractorMOG2()

_, for_track = cam.read()

while True:
    _, frame = cam.read()
    
    contor = frame
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    diff = backSub.apply(frame)
    
    _, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    
    contours, hieararchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    people_list = []
    for c in contours:
        if cv2.contourArea(c) > 1200:            
            people_list.append(cv2.contourArea(c))
            
            hull = cv2.convexHull(c)
            cv2.drawContours(contor, [hull], -1, (255,255,255), 3)
                
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(contor, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(contor, "People", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(contor, "Total People: " + str(len(people_list)), (contor.shape[1] - 400, contor.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("contour", contor)
    
    if cv2.waitKey(33) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cam.release()
        break
    
cam.release()
cv2.destroyAllWindows()