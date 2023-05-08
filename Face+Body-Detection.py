# Emily Dickey Face and Body recognition code for Independent study project
import os
import cv2
import face_recognition
import numpy as np


class faceRec:
    faceLoc = []
    faceEncode = []
    faceNames = []
    knownFaceNames = []
    knownFaceEncode = []
    process_face = True

    def __init__(self):
        self.encodeFace()

    def encodeFace(self):
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            encode = face_recognition.face_encodings(face_image)[0]
            self.knownFaceEncode.append(encode)
            self.knownFaceNames.append(image)

        print(self.knownFaceNames)

    def run_rec(self):
        # Setup camera
        cap = cv2.VideoCapture(0)

        # Initialize detection mode as upper body
        detection_mode = "upper"

        # Initialize face size list
        face_sizes = []

        if self.process_face:
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

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

                        top = int(top / 0.75)
                        right = int(right / 0.75)
                        bottom = int(bottom / 0.75)
                        left = int(left / 0.75)

                        self.faceNames.append(f'{name}')

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                                    1)

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

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = faceRec()
    fr.run_rec()
