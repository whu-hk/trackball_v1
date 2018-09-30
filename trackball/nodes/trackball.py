#!/usr/bin/env python
import rospy
import cv2
import sys
import math
import argparse
import imutils
import time
import numpy as np
import thread

from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
from contour_color.shapedetector import ShapeDetector

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
         
class TrackBall(object):
    def __init__(self, node_name):	      
        self.node_name = node_name
        rospy.init_node(node_name)
        rospy.loginfo("Starting node " + str(node_name))
        rospy.on_shutdown(self.cleanup)
        self.show_text = rospy.get_param("~show_text", True)
        self.ROI = RegionOfInterest()
        self.roi_pub = rospy.Publisher("/roi", RegionOfInterest, queue_size=1)
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None
        self.cps = 0 
        self.cps_values = list()
        self.cps_n_values = 20
        self.radius = 0       
        self.bridge = CvBridge()
        self.track_box = None
        self.has_tracked = False
        self.redLower = None
        self.redUpper = None
        self.old_frame = None
        self.h_min = rospy.get_param("~h_min", 0)
        self.h_max = rospy.get_param("~h_max", 10)	
        self.s_min = rospy.get_param("~s_min", 43)	
        self.s_max = rospy.get_param("~s_max", 255)	
        self.v_min = rospy.get_param("~v_min", 46)	
        self.v_max = rospy.get_param("~v_max", 255)
        self.area_percent = rospy.get_param("~area_percent", 90)	
        self.image_pub = rospy.Publisher("camera/image",Image,queue_size=5)
        self.image_sub = rospy.Subscriber("/camera/bgr/image_raw", Image, self.image_callback, queue_size=1)
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
       	    if tracker_type == 'BOOSTING':
            	self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
            	self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
            	self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
            	self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
            	self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
            	self.tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
            	self.tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
            	self.tracker = cv2.TrackerCSRT_create()
    # These are the callbacks for the slider controls
    def set_h_min(self, pos):
        self.h_min = pos
    def set_h_max(self, pos):
        self.h_max = pos
    def set_s_min(self, pos):
        self.s_min = pos
    def set_s_max(self, pos):
        self.s_max = pos   
    def set_v_min(self, pos):
        self.v_min = pos       
    def set_v_max(self, pos):
       self.v_max = pos
    def set_area_percent(self, pos):
        self.area_percent = pos

    def image_callback(self, data):
        timer = cv2.getTickCount()
        cv2.namedWindow("Parameters")
        cv2.moveWindow("Parameters",0,540)
        # Create the slider controls for HSV
        cv2.createTrackbar("h_min", "Parameters", self.h_min, 255, self.set_h_min)
        cv2.createTrackbar("h_max", "Parameters", self.h_max, 255, self.set_h_max)
        cv2.createTrackbar("s_min", "Parameters", self.s_min, 255, self.set_s_min)
        cv2.createTrackbar("s_max", "Parameters", self.s_max, 255, self.set_s_max)
        cv2.createTrackbar("v_min", "Parameters", self.v_min, 255, self.set_v_min)
        cv2.createTrackbar("v_max", "Parameters", self.v_max, 255, self.set_v_max)
        cv2.createTrackbar("area_percent", "Parameters", self.area_percent, 100,self.set_area_percent)
        
        # Store the image header in a global variable
        self.image_header = data.header
        # Convert the ROS image to OpenCV format using a cv_bridge helper function
        frame = self.convert_image(data)	    
        # Store the frame width and height in a pair of global variables
        if self.frame_width is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.frame_width, self.frame_height = self.frame_size

        #########
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        processed_image = self.process_image(frame)
        if self.has_tracked == True:
            result = self.track_ball(frame)
            if result == True:
                # Publish the tracked ROI
                rospy.loginfo("track success")
                self.publish_track_roi()
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                self.publish_frame1(frame, fps)
            else:
                # track failed,select a new box to track
                rospy.loginfo("track failed")
                cnts = cv2.findContours(processed_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                sd = ShapeDetector()
                flag = False
                contours = []
                # only proceed if at least one contour was found
                if len(cnts) > 0:
                    for c in cnts:
                        shape = sd.detect(c)
                        ((cx, cy), cradius) = cv2.minEnclosingCircle(c)
                        area_per = cv2.contourArea(c) / (math.pi * math.pow(cradius,2))

                        #list conditions all here(cv2.HoughCircles)
                        if shape == "circle" and area_per * 100 >= self.area_percent:
                            flag = True
                            contours.append(c)
                    if flag == True:
                        rospy.loginfo("num_contours: %d", len(contours));
                        cc = max(contours,key=cv2.contourArea)
                        mc = cv2.moments(cc)
                        ((self.x, self.y), self.radius) = cv2.minEnclosingCircle(cc)
                        #TODO
                        self.has_tracked = True
                        self.track_box = cv2.boundingRect(cc) ###
                        try:
                            self.tracker = cv2.TrackerKCF_create()
                            init_ok = self.tracker.init(frame,self.track_box)
                            rospy.loginfo("%s",str(init_ok))
                        except:
                            rospy.loginfo("tracker init failed")
                        rospy.loginfo("after track fail,find a new box to track")
                        self.publish_track_roi()
                        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                        self.publish_frame1(frame, fps)
                    else:
                        rospy.loginfo("after track fail,wait box to detect")
                        self.has_tracked = False
                        self.track_box = None
                        self.publish_roi()
                        self.publish_frame(frame)
                else:
                    rospy.loginfo("after track fail,wait box to detect")
                    self.has_tracked = False
                    self.track_box = None
                    self.publish_roi()
                    self.publish_frame(frame)
        else:
            rospy.loginfo("no tracked before")
            cnts = cv2.findContours(processed_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            sd = ShapeDetector()
            flag = False
            contours = []
            # only proceed if at least one contour was found
            if len(cnts) > 0:
	    	for c in cnts:
	            shape = sd.detect(c)
	            ((cx, cy), cradius) = cv2.minEnclosingCircle(c)
	            area_per = cv2.contourArea(c) / (math.pi * math.pow(cradius,2))
	            #rospy.loginfo("shape: %s", shape);
	            #rospy.loginfo("area_per: %.2f", area_per);
	            #list conditions all here(cv2.HoughCircles)
	            if shape == "circle" and area_per * 100 >= self.area_percent:
	                flag = True
	                contours.append(c)
	        if flag == True:
	            rospy.loginfo("num_contours: %d", len(contours));
	            cc = max(contours,key=cv2.contourArea)
	            mc = cv2.moments(cc)
	            #center = (int(mc["m10"] /mc["m00"]), int(mc["m01"] / mc["m00"]))
	            ((self.x, self.y), self.radius) = cv2.minEnclosingCircle(cc)
	            #TODO
	            self.has_tracked = True
	            self.track_box = cv2.boundingRect(cc)
                    try:
                        self.tracker = cv2.TrackerKCF_create()
                        init_ok = self.tracker.init(frame,self.track_box)
                        rospy.loginfo("%s",str(init_ok))
                    except:
                        rospy.loginfo("tracker init failed")
                    rospy.loginfo("find a new box to track")
	            self.publish_track_roi()
	            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
	            self.publish_frame1(frame, fps)
	        else:
                    rospy.loginfo("wait box to detect")
	            self.has_tracked = False
	            self.track_box = None
	            # Publish the NULL ROI
	            self.publish_roi()
	            self.publish_frame(frame)
            else:
                rospy.loginfo("wait box to detect")
	        self.has_tracked = False
	        self.track_box = None
	        # Publish the NULL ROI
	        self.publish_roi()
	        self.publish_frame(frame)

    def track_ball(self,frame): # frame continue update
        try:
            timer = cv2.getTickCount()
            ok,bbox = self.tracker.update(frame)
            track_fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer)); 
            cv2.putText(frame, tracker_type + " Tracker", (300,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),1);
            if ok:  
                self.has_tracked = True
                self.track_box = bbox
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                return True
            else:
                #self.tracker.release()
                self.has_tracked = False
                self.track_box = None
                return False
        except:
            rospy.loginfo("tracker update failed") 
            return False

    def publish_track_roi(self):
        try:
            roi = RegionOfInterest()
            x,y,w,h = self.track_box
            roi.x_offset = x + 0.5 * w
            roi.y_offset = y + 0.5 * h
            roi.width = w
            roi.height = h
            self.roi_pub.publish(roi)
        except:
            rospy.loginfo("Publishing ROI failed")

    def publish_roi(self):
        try:
            roi = RegionOfInterest()
            roi.x_offset = 0
            roi.y_offset = 0
            roi.width = 0
            roi.height = 0
            self.roi_pub.publish(roi)
        except:
            rospy.loginfo("Publishing ROI failed")

    def publish_frame(self,frame):
        cv2.putText(frame, "No tracking object", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255))
        frame = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_pub.publish(frame)

    def publish_frame1(self,frame,fps):
        x,y,w,h = self.track_box
        cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)
        # Compute the time for this loop and estimate CPS as a running average
        self.cps_values.append(fps)
        if len(self.cps_values) > self.cps_n_values:
            self.cps_values.pop(0)
        self.cps = int(sum(self.cps_values) / len(self.cps_values))
        # Display CPS and image resolution if asked to
        if self.show_text:
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            if self.frame_size[0] >= 640:
                vstart = 25
                voffset = int(50 + self.frame_size[1] / 120.)
            elif self.frame_size[0] == 320:
                vstart = 15
                voffset = int(35 + self.frame_size[1] / 120.)
            else:
                vstart = 10
                voffset = int(20 + self.frame_size[1] / 120.)
            cv2.putText(frame, "CPS: " + str(self.cps), (10, vstart), font_face, font_scale,(0, 0, 255))
            cv2.putText(frame, "RES: " + str(self.frame_size[0]) + "X" + str(self.frame_size[1]), (10, voffset), font_face, font_scale, (0, 0, 255))
        frame2 = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_pub.publish(frame2)

    def convert_image(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            return np.array(cv_image, dtype=np.uint8)
        except CvBridgeError(e):
            print(e)
    
    def process_image(self, image):
        frame = imutils.resize(image, width=640)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "red", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        self.redLower = (self.h_min, self.s_min, self.v_min)
        self.redUpper = (self.h_max, self.s_max, self.v_max)
        mask = cv2.inRange(hsv, self.redLower, self.redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("red_mask",mask)
        cv2.waitKey(1)
        return mask
        
    def cleanup(self):
        print("Shutting down vision node.")
        cv2.destroyAllWindows()       

def main(args):
    try:
        node_name = "trackball"
        TrackBall(node_name)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down trackball node.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
