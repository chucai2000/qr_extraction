import os
import cv2
import zbar # QR decode is based on ZBar library

# ------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------
QR_CODE_PATCH_MIN_SIZE = 150


# ------------------------------------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------------------------------------
def decode_one(img):

    """
    Send input image into ZBar, a third party qr code decoder
    @param img represents the input image in Numpy matrix format
    @return the decoded information
    """

    """ obtain image data"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height = img.shape
    
    """ resize"""
    maxLen = max([width, height])
    if maxLen < QR_CODE_PATCH_MIN_SIZE:
        img = cv2.resize(img, (150, 150))
    else:
        img = cv2.resize(img, (maxLen, maxLen))

    width, height = img.shape
    raw = img.tostring()

    """ create a reader """
    scanner = zbar.ImageScanner()

    """ configure the reader """
    scanner.parse_config('enable')

    """ wrap image data """
    image = zbar.Image(width, height, 'Y800', raw)

    """ scan the image for barcodes """
    scanner.scan(image)

    """ extract results """
    result = None
    for symbol in image:

        """ Just do something useful with results """
        print('Successfully Decoding: ', symbol.type, 'symbol', '"%s"' % symbol.data)
        result = symbol.data
        break

    if result is None:
        print('Failed Decoding ...')

    return result

