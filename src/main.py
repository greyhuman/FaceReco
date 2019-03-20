import numpy as np
#import tensorflow as tf

import sys
import cv2
import dlib
import openface
import face_recognition_models


class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text='models/deploy.prototxt',
                 dnn_model='models/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                #dlib rectangle for alignment
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    @staticmethod
    def draw_marks(image, marks, color=(0, 255, 0)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark.x), int(
                mark.y)), 1, color, -1, cv2.LINE_AA)

def compare_2_faces(known_face_encoding, face_encoding_to_check):
    return (np.linalg.norm(known_face_encoding - face_encoding_to_check))

def compare_faces(known_faces, face_to_check):
    tolerance=0.6
    ind, length = -1, sys.float_info.max
    for i in range(len(known_faces)):
        for face in known_faces[i]:
            cur = compare_2_faces(face, face_to_check)
            if (cur < length):
                length = cur
                ind = i
    if (length <= tolerance):
        return ind
    else:
        return -1


def main():

    classes = ['MXG', 'Sanaken', 'Zofinka', 'Toalk', 'Zissxzirsziiss', 'kiasummer']
    
    known_face_encodes = [
	np.loadtxt('persons/MXG/fv.txt'), 
	np.loadtxt('persons/Sanaken/fv.txt'), 
	np.loadtxt('persons/Zofinka/fv.txt')#,
        #np.loadtxt('persons/Toalk/fv.txt'),
	#np.loadtxt('persons/Zissxzirsziiss/fv.txt'),
	#np.loadtxt('persons/kiasummer/fv.txt')
    ]
    
    #known_face_encodes = np.reshape(known_face_encodes, (6, 5, 128))
    #byke for compare faces
    known_face_encodes = np.reshape(known_face_encodes, (3, 1, 128))

    #get image
    image = cv2.imread('team.jpg', 1)

    #get bboxes
    fd = FaceDetector()
    _, faceboxes = fd.get_faceboxes(image)
    
    #get alignment model
    predictor_model = "models/shape_predictor_68_face_landmarks.dat"
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)
    

    for i in range(len(faceboxes)):
        # print(faceboxes[i][0], faceboxes[i][1], faceboxes[i][2], faceboxes[i][3])
        face_rect = dlib.rectangle(faceboxes[i][0], faceboxes[i][1], faceboxes[i][2], faceboxes[i][3])
        
        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        output = cv2.resize(alignedFace, (faceboxes[i][2] - faceboxes[i][0], faceboxes[i][3] - faceboxes[i][1]))

        #draw marks
        parts = dlib.full_object_detection.parts(pose_landmarks)
        FaceDetector.draw_marks(image, parts)
        #image[faceboxes[i][1]:faceboxes[i][3], faceboxes[i][0]:faceboxes[i][2]] = output


        #init predection model
        predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
        pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)
        face_recognition_model = face_recognition_models.face_recognition_model_location()
        face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


        #get face landmarks for feature extraction
        landmark_set = pose_predictor_5_point(alignedFace, dlib.rectangle(0, 0, alignedFace.shape[0], alignedFace.shape[1]))

        #get feature vector
        feature_vector = np.array(face_encoder.compute_face_descriptor(alignedFace, landmark_set, 1))
        

        #uncomment for adding feature_vector. 'w' for write, 'a' for append
        #with open('persons/Sanaken/fv.txt', 'w') as outfile:
        #    np.savetxt(outfile, feature_vector)

        #known_face_encode = np.loadtxt('persons/MXG/fv.txt')
        ind = compare_faces(known_face_encodes, feature_vector)
        if (ind != -1):
            cv2.putText(image, classes[ind], (faceboxes[i][0], faceboxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(image, "Unknown", (faceboxes[i][0], faceboxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image, (faceboxes[i][0], faceboxes[i][1]), (faceboxes[i][2], faceboxes[i][3]), (255, 0, 0))
    


    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
