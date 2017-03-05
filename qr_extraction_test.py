import os
from qr_extraction import *

def detection_batch():

	""" Test Detection method """

	# Data path
	IMG_PATH = '../../data/barcode-images/warehouses'
	IMG_SAVE_PATH = '../../data/logs'
	IMG_NAMES = [f for f in os.listdir(IMG_PATH) if not os.path.isdir(IMG_PATH+'/'+f)]

	# Display intermediate results or not 
	# The first element controls the pop-up hypothesis and the second element controls the save of intermediate results 
	is_pop_display = (False, False, True)
	
	for IMG_NAME in IMG_NAMES:
		img = cv2.imread(IMG_PATH+'/'+IMG_NAME)
		old_size = img.shape
		if old_size[1] > 1600:
			img = cv2.resize(img,(1600,(int)(1600*old_size[0]/old_size[1])))
	
		print("\n####################################################################")
		print(IMG_NAME + ": Dimension is " + str(img.shape))

		block_mask, block_mask_raw, qr_code_hypothesis_detections = detect_qr_code(img, is_pop_display[0])

		# TEST TEST
		rectify_by_binarization(img, qr_code_hypothesis_detections)
		
		if is_pop_display[1]:
			cv2.imwrite(IMG_SAVE_PATH+'/316-blur15-180-plusAll-morph'+IMG_NAME+'_block_mask.bmp',block_mask)
			cv2.imwrite(IMG_SAVE_PATH+'/316-blur15-180-plusAll-morph'+IMG_NAME+'_block_mask_raw.bmp',block_mask_raw)
	
		if is_pop_display[2]:
			img_draw = img
			for detection in qr_code_hypothesis_detections:
				img_draw = visualize_detections(img_draw, detection)
			cv2.imwrite(IMG_SAVE_PATH+'/'+IMG_NAME+'_img_patch'+'.bmp',img_draw)


def decoding_batch():
	
	""" Test Decoding Method or Library """

	# Data path
	IMG_PATH = '../../data/barcode-images/decoding'
	IMG_NAMES = os.listdir(IMG_PATH)
	for IMG_NAME in IMG_NAMES:

		img_qr_patch = cv2.imread(IMG_PATH+'/'+IMG_NAME)

		displays1((img_qr_patch,))
		decode_result = decode_one(img_qr_patch)
		print(decode_result)


def detection_and_decoding_batch():

	""" Test both Detection and Decode method """

	# Data path
	IMG_PATH = '../../data/barcode-images/warehouses'
	IMG_SAVE_PATH = '../../data/logs'
	IMG_NAMES = [f for f in os.listdir(IMG_PATH) if not os.path.isdir(IMG_PATH+'/'+f)]
	
	# Display intermediate results or not 
	# The first element controls the pop-up hypothesis and the second element controls the save of intermediate results 
	is_camera_id_prefix=True
	is_pop_display = (False, False, False)
	dict_station = {	'ws-rec-14-04':set(), 
					'ws-rec-14-05':set(), 
					'ws-rec-16-01':set(), 
					'ws-rec-16-03':set(), 
					'ws-rec-16-04':set(), 
					'ws-rec-16-05':set(), 
					'ws-rec-16-06':set(), 
					'ws-rec-16-07':set(), 
					'ws-rec-16-12':set()
					}

	for IMG_NAME in IMG_NAMES:

		if is_camera_id_prefix:
			camera_id = IMG_NAME[:-11] # revised according to the naming rule of the retrieved image from python script
		else:
			camera_id = 'camera_id'

		decode_results = detect_and_decode_qr_code(IMG_PATH+'/'+IMG_NAME,
						camera_id)

		""" Check whether the decode result hits a ground truth """
		for decode_result in decode_results:
			if decode_result in dict_station.keys():
				dict_station[decode_result].add(camera_id)

	print('\n\n------ The Result of QR Extraction at this round ------')
	print(dict_station)
	print('--- with hit rate: ')
	print(qr_extraction_hit_rate(dict_station))


if __name__ == '__main__':

	""" Detection only """
	#detection_batch()

	""" Decoding only """
	#decoding_batch()

	""" Detection and Decoding """
	detection_and_decoding_batch()



