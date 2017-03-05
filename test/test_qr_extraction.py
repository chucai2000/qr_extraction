import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..')))
from qr_extraction import *


class TestQRExtraction(unittest.TestCase):

    def test_detection_batch(self):

        """ Test Detection method """

        # Data path
        img_path = 'resources/input_detection'
        img_save_path = 'resources/output'
        img_names = [f for f in os.listdir(img_path) if not os.path.isdir(img_path+'/'+f)]

        # Display intermediate results or not
        # The first element controls the pop-up hypothesis and
        # the second element controls the save of intermediate results
        is_pop_display = (False, False, True)

        for img_name in img_names:
            img = cv2.imread(img_path+'/'+img_name)
            old_size = img.shape
            if old_size[1] > 1600:
                img = cv2.resize(img, (1600, int(1600*old_size[0]/old_size[1])))

            print("\n####################################################################")
            print(img_name + ": Dimension is " + str(img.shape))

            block_mask, block_mask_raw, qr_code_hypothesis_detections = detect_qr_code(img, is_pop_display[0])

            # TEST TEST
            rectify_by_binarization(img, qr_code_hypothesis_detections)

            if is_pop_display[1]:
                cv2.imwrite(img_save_path+'/316-blur15-180-plusAll-morph'+img_name+'_block_mask.bmp',
                            block_mask)
                cv2.imwrite(img_save_path+'/316-blur15-180-plusAll-morph'+img_name+'_block_mask_raw.bmp',
                            block_mask_raw)

            if is_pop_display[2]:
                img_draw = img
                for detection in qr_code_hypothesis_detections:
                    img_draw = visualize_detections(img_draw, detection)
                cv2.imwrite(img_save_path+'/'+img_name+'_img_patch'+'.bmp', img_draw)

    def test_decoding_batch(self):

        """ Test Decoding Method or Library
        zbar library is required to start qr decode.
        """


        # Data path
        img_path = 'resources/input_decode'
        img_names = os.listdir(img_path)
        for img_name in img_names:

            img_qr_patch = cv2.imread(img_path+'/'+img_name)

            displays1((img_qr_patch,))
            decode_result = decode_one(img_qr_patch)
            print(decode_result)


if __name__ == '__main__':
    unittest.main()



