import numpy as np
#import tensorflow as tf

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

def compare_faces(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    return (np.linalg.norm(known_face_encoding - face_encoding_to_check) <= tolerance)


def main():
    

    #hardcoded MXG face
    known_face_encode = [-1.26654908e-01,  1.35700598e-01,  1.53716467e-02, -8.24333429e-02,
 -1.72865108e-01, -4.38526198e-02,  3.02785020e-02, -1.34015307e-01,
  1.84273794e-01, -1.17534235e-01,  2.33109221e-01, -6.04636520e-02,
 -3.26556951e-01, -5.96866831e-02,  5.54038584e-02,  1.85909823e-01,
 -2.09557459e-01, -1.04872480e-01, -1.08147666e-01, -1.13650486e-01,
 -3.24867070e-02,  4.29936312e-02,  1.54815083e-02,  7.45417476e-02,
 -1.35438249e-01, -2.85149753e-01, -5.89919873e-02, -1.11097962e-01,
 -1.18976608e-02, -1.17783517e-01,  2.78510582e-02,  1.13265410e-01,
 -2.00322587e-02, -7.42890686e-02,  3.96273658e-02, -1.02639198e-04,
 -9.75040793e-02, -3.41068804e-02,  2.84794778e-01,  7.05836490e-02,
 -1.99088544e-01,  2.05509225e-03,  5.24227470e-02,  3.08449954e-01,
  2.52037525e-01,  4.02835244e-03,  3.89107615e-02, -7.58150220e-02,
  2.00045198e-01, -2.82067537e-01,  3.00010629e-02,  1.10228591e-01,
  1.33863911e-01,  6.27712309e-02,  9.25320089e-02, -1.86661139e-01,
 -2.92366948e-02,  1.58955529e-01, -2.54328579e-01,  5.12454621e-02,
  1.40671641e-01, -1.60912395e-01, -4.95279171e-02, -3.32753658e-02,
  2.25689799e-01,  1.32995740e-01, -1.41886607e-01, -1.11932442e-01,
  1.61068514e-01, -1.54709786e-01, -1.69450283e-01,  4.91478071e-02,
 -1.20110884e-01, -1.70313627e-01, -2.82522678e-01,  2.76208222e-02,
  3.06639582e-01,  1.73616409e-01, -1.53296828e-01, -3.63254994e-02,
  2.60761846e-03, -2.81884279e-02,  4.27182950e-02,  1.28632694e-01,
 -1.09648637e-01, -8.40880126e-02, -1.72065962e-02, -2.99107470e-02,
  1.30687252e-01, -5.21629117e-03, -6.32341057e-02,  2.90156811e-01,
  5.61903529e-02,  6.05171248e-02,  4.77227196e-02, -9.91216674e-03,
 -1.06739029e-01, -3.90159227e-02, -1.24813974e-01, -3.38059515e-02,
  2.07611118e-02, -1.06981397e-01,  7.45271146e-02,  5.04177064e-02,
 -1.94249824e-01,  2.70209730e-01, -4.38959226e-02,  3.87569070e-02,
 -1.10202923e-01, -5.04918061e-02, -4.38217409e-02,  1.22803733e-01,
  1.82470426e-01, -2.57067412e-01,  2.08032280e-01,  1.64141729e-01,
  5.72616048e-02,  1.47231609e-01,  8.82415324e-02, -4.17935513e-02,
  6.58648508e-03, -1.05878552e-02, -1.33494303e-01, -8.07538554e-02,
  1.12990327e-01, -6.33426905e-02,  1.43287987e-01,  5.97111546e-02]

    #get image
    image = cv2.imread('1.jpg', 1)

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
        #print(feature_vector)

        if (compare_faces(known_face_encode, feature_vector)):
            cv2.putText(image, "MXG", (faceboxes[i][0], faceboxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(image, "Unknown", (faceboxes[i][0], faceboxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(image, (faceboxes[i][0], faceboxes[i][1]), (faceboxes[i][2], faceboxes[i][3]), (255, 0, 0))
    





    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
