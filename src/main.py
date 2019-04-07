import numpy as np
# import tensorflow as tf

import os
import time
import sys
import cv2
import dlib
import openface
import face_recognition_models

MAIN_PATH = os.path.dirname(os.path.realpath(__file__))


class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text=MAIN_PATH + '/models/deploy.prototxt',
                 dnn_model=MAIN_PATH + '/models/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

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
                # dlib rectangle for alignment
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        fboxes = []
        for box in faceboxes:
            # Move box down.
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                fboxes.append(facebox)

        return confidences, fboxes

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
    tolerance = 0.6
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


def main(mode='test', img_path='def'):
    t = time.clock()
    classes = ['MXG', 'Sanaken', 'Zofinka', 'Toalk', 'Zissxzirsziiss', 'kiasummer']

    known_face_encodes = [
        np.loadtxt(MAIN_PATH + '/persons/MXG/fv.txt'),
        np.loadtxt(MAIN_PATH + '/persons/Sanaken/fv.txt'),
        np.loadtxt(MAIN_PATH + '/persons/Zofinka/fv.txt'),
        np.loadtxt(MAIN_PATH + '/persons/Toalk/fv.txt'),
        np.loadtxt(MAIN_PATH + '/persons/Zissxzirsziiss/fv.txt'),
        np.loadtxt(MAIN_PATH + '/persons/kiasummer/fv.txt')
    ]

    known_face_encodes = np.reshape(known_face_encodes, (6, 5, 128))

    # get image
    if img_path == 'def':
        image = cv2.imread('team.jpg', 1)
    else:
        image = cv2.imread(img_path, 1)

    # output
    bbox_mark_image = image.copy()
    init_align_faces = []
    out_arr = []

    # get bboxes
    fd = FaceDetector()
    conf, faceboxes = fd.get_faceboxes(image)

    # get alignment model
    predictor_model = MAIN_PATH + "/models/shape_predictor_68_face_landmarks.dat"

    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    # init predection model
    predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
    pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)
    face_recognition_model = face_recognition_models.face_recognition_model_location()
    face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

    for i in range(len(faceboxes)):
        # get dlib rectangle from facebox
        face_rect = dlib.rectangle(faceboxes[i][0], faceboxes[i][1], faceboxes[i][2], faceboxes[i][3])

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        alignedFace_out = cv2.resize(alignedFace,
                                     (faceboxes[i][2] - faceboxes[i][0], faceboxes[i][3] - faceboxes[i][1]))
        initFace_out = image[faceboxes[i][1]:faceboxes[i][3], faceboxes[i][0]:faceboxes[i][2]]
        init_align_faces.append([initFace_out, alignedFace_out])

        # draw marks
        parts = dlib.full_object_detection.parts(pose_landmarks)
        FaceDetector.draw_marks(bbox_mark_image, parts)

        # get face landmarks for feature extraction
        landmark_set = pose_predictor_5_point(alignedFace,
                                              dlib.rectangle(0, 0, alignedFace.shape[0], alignedFace.shape[1]))

        # get feature vector
        feature_vector = np.array(face_encoder.compute_face_descriptor(alignedFace, landmark_set, 1))

        # known_face_encode = np.loadtxt('persons/MXG/fv.txt')
        ind = compare_faces(known_face_encodes, feature_vector)
        if (ind != -1):
            face_class = classes[ind]
            colour = (0, 255, 0)
        else:
            face_class = "Unknown"
            colour = (0, 0, 255)
        cv2.putText(image, face_class, (faceboxes[i][0], faceboxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour,
                    1)
        cv2.rectangle(bbox_mark_image, (faceboxes[i][0], faceboxes[i][1]), (faceboxes[i][2], faceboxes[i][3]),
                      (0, 255, 0))
        cv2.rectangle(image, (faceboxes[i][0], faceboxes[i][1]), (faceboxes[i][2], faceboxes[i][3]), (0, 255, 0))

        out_arr.append({'x1': faceboxes[i][0], 'y1': faceboxes[i][1], 'x2': faceboxes[i][2], 'y2': faceboxes[i][3],
                        'class': face_class})

    t = time.clock() - t
    out_imgs = [image, bbox_mark_image, init_align_faces, t]
    if mode == 'test':
        cv2.imshow("Preview", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if mode == 'metr':
        return out_arr
    if mode == 'process':
        return out_imgs
    if mode == 'def':
        return out_imgs


if __name__ == '__main__':
    main()
