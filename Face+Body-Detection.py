# Emily Dickey Face and Body recognition code for Independent study project

"""The program loads images of known faces from the "Faces" directory and encodes
them. Then, it uses OpenCV to capture live video from the default camera and detect faces and bodies in each frame."""

import os
import cv2
import face_recognition
import numpy as np


class faceRec:
    faceLoc = []  # A list to store the locations of detected faces
    faceEncode = []  # A list to store the face encodings of detected faces
    faceNames = []  # A list to store the names of detected faces
    knownFaceNames = []  # A list to store the names of known faces
    knownFaceEncode = []  # A list to store the face encodings of known faces
    process_face = True  # A flag to enable/disable face recognition

    def __init__(self):
        self.encodeFace() # Call the encodeFace method on initialization

    """face_recognition.face_locations() to locate faces 
in the frame and face_recognition.face_encodings() to generate 
a face encoding for each detected face. """
    def encodeFace(self):
        for image in os.listdir('Faces'): # Loop through all images in the 'Faces' directory
            face_image = face_recognition.load_image_file(f'Faces/{image}')  # Load the image using face_recognition library
            encode = face_recognition.face_encodings(face_image)[0] # Encode the face in the image
            self.knownFaceEncode.append(encode) # Add the face encoding to the known face encodings list
            self.knownFaceNames.append(image)  # Add the image name to the known face names list


        print(self.knownFaceNames) # Print out the list of known face names

    def run_rec(self):
        # Setup camera
        cap = cv2.VideoCapture(0) # Create a VideoCapture object to capture frames from the default camera (index 0)

        # Initialize detection mode as upper body
        detection_mode = "upper"

        # Initialize face size list
        face_sizes = []

        if self.process_face: # If face recognition is enabled
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()  # Read a frame from the camera

                # Show the captured image
                self.faceNames = [] # Reset the face names list

                # Resize frame
                smallFrame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)  # Resize the frame to 75% of its original size
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale

                # Detect faces in current frame
                face_locations = face_recognition.face_locations(smallFrame) # Locate faces in the resized frame
                face_encodings = face_recognition.face_encodings(smallFrame, face_locations)# Encode the faces in the resized frame


                # Draw bounding box around the face and identify name associated
                for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
                    match = face_recognition.compare_faces(self.knownFaceEncode, face_encodings) # Compare the detected face encoding with known face encodings
                    name = 'Unknown'  # Default name is 'Unknown'
                    name = 'Unknown'

                    face_distances = face_recognition.face_distance(self.knownFaceEncode, face_encodings)
                    best_match = np.argmin(face_distances)  # Get the index of the best matching known face


                    if match[best_match]:  # If the detected face matches a known face
                        name = self.knownFaceNames[best_match] # Get the name of the known face

                        top = int(top / 0.75)
                        right = int(right / 0.75)
                        bottom = int(bottom / 0.75)
                        left = int(left / 0.75)

                        # Add name of recognized face to list
                        self.faceNames.append(f'{name}')

                        # Draw a rectangle around the face and add name label
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                                    1)

                if detection_mode == "upper":
                    # Detect upper bodies in the frame
                    body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
                    bodies = body_cascade.detectMultiScale(frame, 1.1, 5)

                    if len(face_locations) > 0:
                        # Calculate the distance from the top of the image to the top of the face
                        face_distance = face_locations[0][0] / smallFrame.shape[0]
                        if face_distance < 0.4:
                            # Detect full bodies in the frame
                            detection_mode = "full"
                else:
                    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
                    bodies = body_cascade.detectMultiScale(frame, 1.1, 5)
                    # Check if a face was detected near the top of the image and switch to upper body detection mode
                    if len(face_locations) > 0:
                        # Calculate the distance from the top of the image to the top of the face
                        face_distance = face_locations[0][0] / smallFrame.shape[0]
                        if face_distance > 0.6:
                            detection_mode = "upper"

                    # Process detected bodies
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)

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

                if cv2.waitKey(1) & 0xFF == ord('q'): #exists when the user presses the q key
                    break

            cap.release() #Releases the video capture resourcesn
            cv2.destroyAllWindows() #closes all windows created by OpenCV


if __name__ == '__main__':
    fr = faceRec()
    fr.run_rec() #starts program
