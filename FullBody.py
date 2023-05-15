import os
import cv2
import face_recognition



class FaceRec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_face = True
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(os.path.splitext(image)[0])

    def run_rec(self):
        cap = cv2.VideoCapture(0)

        full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

        while True:
            ret, frame = cap.read()

            face_names = []
            if self.process_face:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]

                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    face_names.append(name)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = full_body_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRec()
    fr.run_rec() #starts program
