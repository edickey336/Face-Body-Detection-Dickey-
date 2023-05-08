import cv2
import face_recognition
import os

class faceRec:
    faceLoc = []
    faceEncode = []
    faceNames = []
    knownFaceNames = []
    knownFaceEncode = []
    process_face = True

    def __init__(self):
        # Load body cascades
        self.body_cascade_full = cv2.CascadeClassifier('haarcascade_fullbody.xml')
        self.body_cascade_upper = cv2.CascadeClassifier('haarcascade_upperbody.xml')

    def encodeFace(self):
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            face_encodings = face_recognition.face_encodings(face_image)
            if len(face_encodings) > 0:
                encode = face_encodings[0]
                self.knownFaceEncode.append(encode)
                self.knownFaceNames.append(os.path.splitext(image)[0])
            else:
                print(f"No faces detected in {image}")
        print(self.knownFaceNames)


    def detect_faces(self, frame):
        # Detect faces in current image
        face_locations = face_recognition.face_locations(frame)
        face_encoding = face_recognition.face_encodings(frame, face_locations)

        # Draw bounding box around the face and identify name associated
        for (top, right, bottom, left), face_encodings in zip(face_locations, face_encoding):
            name = 'Unknown'

            matches = face_recognition.compare_faces(self.knownFaceEncode, face_encodings)
            if True in matches:
                index = matches.index(True)
                name = self.knownFaceNames[index]

            top = int(top)
            right = int(right)
            bottom = int(bottom)
            left = int(left)

            self.faceNames.append(f'{name}')

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
            cv2.rectangle(frame, (left, bottom - 5), (right, bottom), (0, 0, 255), -1)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))

        return frame

    def detect_bodies(self, frame):
        # Detect full body
        bodies_full = self.detect_objectsFull(frame, self.body_cascade_full)

        # Detect upper body
        bodies_upper = self.detect_objectsUpper(frame, self.body_cascade_upper)

        for (x, y, w, h) in bodies_upper:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,235,205), 2)

        for (x, y, w, h) in bodies_full:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        return frame

    def detect_objectsFull(self, frame, classifier):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = classifier.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=4) #between 2-5
        return objects

    def detect_objectsUpper(self, frame, classifier):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) #between 3-6
        return objects

    # def evaluate_performance(self, objects, ground_truth):
    #     # Calculate the intersection over union (IoU) for each detected object
    #     iou_scores = []
    #     for obj in objects:
    #         iou_scores.append([self.calculate_iou(obj, gt) for gt in ground_truth])
    #
    #     # Calculate the maximum IoU for each ground truth object
    #     max_iou_scores = np.max(iou_scores, axis=0)
    #
    #     # Calculate the average IoU over all ground truth objects
    #     avg_iou_score = np.mean(max_iou_scores)
    #
    #     return avg_iou_score
    #
    # def calculate_iou(self, box1, box2):
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2
    #
    #     # Calculate intersection coordinates and area
    #     intersection_x1 = max(x1, x2)
    #     intersection_y1 = max(y1, y2)
    #     intersection_x2 = min(x1 + w1, x2 + w2)
    #     intersection_y2 = min(y1 + h1, y2 + h2)
    #     intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    #
    #     # Calculate union area
    #     box1_area = w1 * h1
    #     box2_area = w2 * h2
    #     union_area = box1_area + box2_area - intersection_area
    #
    #     # Calculate IoU score
    #     iou_score = intersection_area / union_area
    #
    #     return iou_score

    def run_rec(self):
        # Load image
        # from PIL import Image
        # frame =Image.open("C:\pythonProject\DickeyDrone_Final\Faces\Eileen.jpg")
        frame = cv2.imread("C:\pythonProject\DickeyDrone_Final\Faces\Suze.jpeg")

        frame = cv2.resize(frame,(650,900))
        self.encodeFace()

        if self.process_face:
            frame = self.detect_faces(frame)
        frame = self.detect_bodies(frame)

        # Display image
        cv2.imshow('Face Recognition and Body Detection', frame)
        cv2.waitKey(0)


if __name__ == '__main__':
    fr = faceRec()
    fr.run_rec()

