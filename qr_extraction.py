from qr_detection import *
from qr_decode import *


def detect_and_decode_qr_code(img_path, sample_capture_id):

    """ Entry function of QR Detection and Decode """

    img = cv2.imread(img_path)

    old_size = img.shape
    if old_size[1] > 1600:
        img = cv2.resize(img, (1600, int(1600*old_size[0]/old_size[1])))

    print("\n------ Loaded an Image ------")
    print(os.path.basename(img_path) + ": Dimension is " + str(img.shape))
    print('    ... retrieved from [%s]' % sample_capture_id)

    block_mask, block_mask_raw, qr_code_hypothesis_detections = detect_qr_code(img, False)

    """
    Extract the image patches corresponding to 'True' or 'Probable' qr-codes
    """

    decode_results = []

    for detection in qr_code_hypothesis_detections:
        is_qr_code = detection['is_containing_qr_code']
        if is_qr_code is 'True' or is_qr_code is 'Probable':
            img_patch = detection['img_patch']
            decode_result = decode_one(img_patch)

            """ Obtain a list of decoding results, since some image may have multiple qr codes """
            decode_results.append(decode_result)

    print('--- Report list of decoding results ---')
    print(decode_results)

    return decode_results



