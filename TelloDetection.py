import os
import cv2
import face_recognition
import numpy as np
from tello_drone import TelloDrone
import math
"""These are import statements for various modules and libraries that are needed 
in this script. For example, os module is used for operating system related tasks, 
cv2 module is used for image processing, face_recognition module is used for face recognition"""

"""The code is an implementation of a face recognition program that is designed 
to track the faces of individuals by drawing bounding boxes around the faces and identifying 
the names associated with them. It is also designed to track and follow a particular 
target face using a Tello drone."""


class faceRec:
    # Initialize class variables
    faceLoc = []  # list to store face locations
    faceEncode = []  # list to store face encodings
    faceNames = []  # list to store face names
    knownFaceNames = []  # list to store known face names
    knownFaceEncode = []  # list to store known face encodings
    process_face = True  # flag to indicate if face processing is enabled
    target = "unknown"  # target name to track

    def __init__(self):
        self.encodeFace()

    def encodeFace(self):
        # Loop through images in Faces folder and encode faces
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            encode = face_recognition.face_encodings(face_image)[0]
            self.knownFaceEncode.append(encode)
            self.knownFaceNames.append(image)

        print(self.knownFaceNames)

    def run_rec(self):
        # Setup Tello drone
        drone = TelloDrone()
        drone.connect()
        drone.stream_on()

        # Initialize detection mode as upper body
        detection_mode = "upper"

        # Initialize face size list
        face_sizes = []

        # Initialize target name and tracking distance
        target = "Unknown"

        if self.process_face:
            while True:
                # Capture frame-by-frame
                frame = drone.read()

                # Show the captured image
                self.faceNames = []

                # Resize frame
                smallFrame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in current frame
                face_locations = face_recognition.face_locations(smallFrame)
                face_encodings = face_recognition.face_encodings(smallFrame, face_locations)

                # Draw bounding box around the face and identify name associated
                for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
                    match = face_recognition.compare_faces(self.knownFaceEncode, face_encodings)
                    name = 'Unknown'

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
                        self.faceNames.append(f'{name}')

                        # Draw bounding box and text on frame
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                                    1)

                        # Check if detected face is the target person
                        if name == "target":
                            # Calculate distance and move drone towards target
                            face_center_x = (left + right) / 2
                            face_center_y = (top + bottom) / 2
                            frame_center_x = smallFrame.shape[1] / 2
                            frame_center_y = smallFrame.shape[0] / 2
                            x_diff = frame_center_x - face_center_x
                            y_diff = frame_center_y - face_center_y
                            distance = math.sqrt(x_diff**2 + y_diff**2)

                            # Move left or right to center face
                            if abs(x_diff) > 15:
                                if x_diff > 0:
                                    drone.move_left(15)
                                else:
                                    drone.move_right(15)

                            # Move up or down to center face
                            if abs(y_diff) > 15:
                                if y_diff > 0:
                                    drone.move_down(15)
                                else:
                                    drone.move_up(15)

                            # Move forward or backward to adjust distance
                            if distance < 0.3:
                                drone.move_forward(15)
                            elif distance > 0.35:
                                drone.move_backward(15)

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

                    # Get the coordinates of the face detected in the full body detection
                    full_body_face_coords = None
                    for box in bodies:
                        if box[4] == "face":
                            full_body_face_coords = box[:4]
                            break

                    # Get the coordinates of the face detected in the upper body detection
                    upper_body_face_coords = None
                    for box in bodies:
                        if box[4] == "face":
                            upper_body_face_coords = box[:4]
                            break

                    # Compare the coordinates to check if it's the same face
                    if full_body_face_coords is not None and upper_body_face_coords is not None:
                        if full_body_face_coords == upper_body_face_coords:
                            target = "target"
                    else:
                        target = None


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
