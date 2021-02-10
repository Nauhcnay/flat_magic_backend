import argparse
from os.path import *
import os
import subprocess
import shutil
import time
import pdb

def parse():
	# path to executable file
	p1 = normcase("./")
	p2 = 'polyvector_thing.exe'
	# path to input files
	p3 = normcase("E:\\OneDrive - George Mason University\\00.Projects\\01.Sketch cleanup\\00.Bechmark Dataset\\SSEB Dataset with GT")
	# path to output files, if necessary
	# p4 = normcase("./results")
	p4 = None
	parser = argparse.ArgumentParser(description='Batch Run Script')
	parser.add_argument('--exe', 
                    help ='path to executable file',
                    default = join(p1,p2))
	parser.add_argument('--input',
					help='path to input files',
					default = p3)
	parser.add_argument('--result',
					help='path where reuslts will be saved, if necessary',
					default = p4)

	return parser

def main():
	args = parse().parse_args()
	for img in os.listdir(args.input):
		name, extension = splitext(img)
		
		if extension == '.png':
			subprocess.run([args.exe, "-noisy", join(args.input, img)])
		
			
			# 这个可能用的到，也可能用不到，以后再详细想想怎么写成一个通用的框架
			# path_to_result = join(args.result, name)
			# time.sleep(1)
			# if not exists(args.result):
			# 	os.mkdir(args.result)
			# if not exists(path_to_result):
			# 	os.mkdir(path_to_result)
			
			# for svg in os.listdir(normcase('./')):
			# 	if svg.endswith('.svg'):
			# 		# pdb.set_trace()
			# 		shutil.move(join(normcase('./'), svg),
			# 			join(path_to_result,svg))
	print("Done")

if __name__ == "__main__":
	main()
