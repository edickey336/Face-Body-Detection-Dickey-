"""These are import statements for various modules and libraries that are needed
in this script. For example, os module is used for operating system related tasks,
cv2 module is used for image processing, face_recognition module is used for face recognition"""

import os
import cv2
import face_recognition
import numpy as np
from tello_drone import TelloDrone
import math

"""The code is an implementation of a face recognition program that is designed 
to track the faces of individuals by drawing bounding boxes around the faces and identifying 
the names associated with them. It is also designed to track and follow a particular 
target face using a Tello drone."""


class faceRec:
    # Initialize class variables
    faceLoc = []  # list to store face locations
    faceEncode = []  # list to store face encodings
    knownFaceNames = []  # list to store known face names
    knownFaceEncode = []  # list to store known face encodings
    process_face = True  # flag to indicate if face processing is enabled
    target = "unknown"  # target name to track

    def __init__(self):
        self.encodeFace()

    def encodeFace(self):
        # Loop through images in Faces folder and encode faces
        print("Encoding Faces...This may take a minute\n")
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            encode = face_recognition.face_encodings(face_image)[0]
            self.knownFaceEncode.append(encode)
            self.knownFaceNames.append(image)

        print("Here are the known faces: ")
        print(self.knownFaceNames)

    def aim_at(drone, point, resolution):
        # Calculate the field of view of the drone's camera.
        fov = 82.6

        # Calculate the angular delta for each pixel in the horizontal direction.
        x_ang_delta = fov / math.hypot(resolution[0], resolution[1])

        # Calculate the angle by which the drone needs to rotate in order to point at the target point.
        center_x = resolution[0] / 2
        target_x = point[0]
        rot_angle = (target_x - center_x) * x_ang_delta

        # Rotate the drone to aim at the target point.
        if target_x < center_x:
            drone.rotate_ccw(15)  # Rotate counter-clockwise.
        else:
            drone.rotate_cw(15)  # Rotate clockwise.

    def run_rec(self):
        # Setup Tello drone
        drone = TelloDrone()
        input("Press Enter to connect to drone.")
        # Connect to drone
        drone.connect(5)
        if not drone.connected:
            print("Trouble connecting to drone.")
            return
        print("Remaining Battery:", drone.last_state['bat'])

        # Initialize face size list
        face_sizes = []
        cap = cv2.VideoCapture(0)
        detection_mode = "upper"

        while True:
            # Take off the drone
            drone.takeoff()

            # Rotate the drone to search for the target
            drone.left(15)

            while True:
                frame = drone.get_frame()
                smallFrame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                cv2.imshow("Tello Video Stream", smallFrame)
                cv2.waitKey(1)

                smallFrame = cv2.resize(frame, (0, 0), fx=0.75,
                                        fy=0.75)  # Resize the frame to 75% of its original size

                face_locations = face_recognition.face_locations(smallFrame)
                face_encodings = face_recognition.face_encodings(smallFrame, face_locations)

                # If a face is detected, then stop rotating the drone.
                if len(face_locations) > 0:
                    break

                if smallFrame is not None:
                    # Draw bounding box around the face and identify name associated
                    for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
                        match = face_recognition.compare_faces(self.knownFaceEncode, face_encodings[0])

                        face_distances = face_recognition.face_distance(self.knownFaceEncode, face_encodings)
                        best_match = np.argmin(face_distances)

                        if match[best_match]:
                            name = self.knownFaceNames[best_match]

                            # Scale bounding box coordinates back to original size
                            top = int(top / 0.75)
                            right = int(right / 0.75)
                            bottom = int(bottom / 0.75)
                            left = int(left / 0.75)

                            # Append name to list of detected face names
                            faceRec.faceNames.append(name)

                            # Draw bounding box and text on frame
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                        (255, 255, 255),
                                        1)

                            # Check if detected face is the target person
                            if name == self.target:
                                # Calculate distance and move drone towards target
                                face_center_x = (left + right) / 2
                                face_center_y = (top + bottom) / 2
                                frame_center_x = smallFrame.shape[1] / 2
                                frame_center_y = smallFrame.shape[0] / 2
                                x_diff = frame_center_x - face_center_x
                                y_diff = face_center_y - frame_center_y  # flip y-axis direction
                                distance = math.sqrt(x_diff ** 2 + y_diff ** 2)

                                # Adjust drone orientation to aim at the target face
                                self.aim_at(drone, (face_center_x, face_center_y), smallFrame.shape[:2])

                                # Move drone towards target face
                                if distance > 100:  # arbitrary distance threshold
                                    if y_diff > 0:
                                        drone.move_up(20)
                                    elif y_diff < 0:

                                        # Move left or right to center face
                                        if abs(x_diff) > 15:
                                            if x_diff > 0:
                                                drone.left(15)
                                            else:
                                                drone.right(15)

                                        # Move up or down to center face
                                        if abs(y_diff) > 15:
                                            if y_diff > 0:
                                                drone.down(15)
                                            else:
                                                drone.up(15)

                                # Move forward or backward to adjust distance
                                if distance < 0.3:
                                    drone.forward(15)
                                elif distance > 0.35:
                                    drone.backward(15)

                # Switch detection mode based on face distance from top of image
                    if detection_mode == "upper":
                        body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
                        bodies = body_cascade.detectMultiScale(frame, 1.1, 5)

                        if len(face_locations) > 0:
                        # Calculate the distance from the top of the image to the top of the face
                            face_distance = face_locations[0][0] / smallFrame.shape[0]
                            if face_distance < 0.4:
                                detection_mode = "full"
                    else:
                        body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
                        bodies = body_cascade.detectMultiScale(frame, 1.1, 5)

                        if len(face_locations) > 0:
                        # Calculate the distance from the top of the image to the top of the face
                            face_distance = face_locations[0][0] / smallFrame.shape[0]
                            if face_distance > 0.6:
                                detection_mode = "upper"

                    # Process detected bodies
                    for (x, y, w, h) in bodies:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)

                    # # Get the coordinates of the face detected in the full body detection
                    #     full_body_face_coords = None
                    #     for box in bodies:
                    #         if box[5] == "face":
                    #             full_body_face_coords = box[:5]
                    #             break
                    #
                    # # Get the coordinates of the face detected in the upper body detection
                    #     upper_body_face_coords = None
                    #     for box in bodies:
                    #         if box[5] == "face":
                    #             upper_body_face_coords = box[:5]
                    #             break
                    #
                    # # Compare the coordinates to check if it's the same face
                    #     if full_body_face_coords is not None and upper_body_face_coords is not None:
                    #         if full_body_face_coords == upper_body_face_coords:
                    #             target == "target"
                    #     else:
                    #         target = None

                # Calculate face size and add to list
                    if len(face_locations) > 0:
                        face_location = face_locations[0]
                        face_size = (face_location[1] - face_location[3]) * (face_location[0] - face_location[2])
                        face_sizes.append(face_size)

                # Switch detection mode based on average face size
                    if len(face_locations) > 0:
                        # Calculate the distance from the top of the image to the top of the face
                        face_distance = face_locations[0][0] / smallFrame.shape[0]
                        if face_distance < 0.4:
                            detection_mode = "full"

                    cv2.imshow('Face and Body Recognition', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):  # exists when the user presses the q key
                        break

                cap.release()  # Releases the video capture resourcesn
                cv2.destroyAllWindows()  # closes all windows created by OpenCV

if __name__ == '__main__':
    fr = faceRec()
    fr.run_rec()  # starts program
