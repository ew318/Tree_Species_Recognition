# Python code for splitting out the images into test/train/validation sets
import os
import numpy as np
import shutil


def make_test_train_validation_dirs(dirname):
	os.mkdir(dirname)
	os.mkdir(dirname + '/train')
	os.mkdir(dirname + '/test')
	os.mkdir(dirname + '/validation')


def get_train_test_validate_dirs(cwd, image_type):
	train_dir = cwd + '/' + image_type + '/train/' + category
	test_dir = cwd + '/' + image_type + '/test/' + category
	validation_dir = cwd + '/' + image_type + '/validation/' + category
	if not os.path.exists(train_dir):
		os.mkdir(train_dir)
	if not os.path.exists(test_dir):
		os.mkdir(test_dir)
	if not os.path.exists(validation_dir):
		os.mkdir(validation_dir)
	return train_dir, validation_dir, test_dir


def assign_train_test_validate(file_path, test_dir, train_dir, validation_dir):
	split_choice = np.random.rand(1)
	if split_choice < 0.7:
		shutil.copy(file_path, train_dir)
	elif split_choice < 0.9:
		shutil.copy(file_path, validation_dir)
	else:
		shutil.copy(file_path, test_dir)


make_test_train_validation_dirs('split_aerial')

for subdir, dirs, files in os.walk('camden_images'):
	for name in files:
		file_path = (os.path.join(subdir, name))
		cat_tree = file_path.split('/')
		if len(cat_tree) == 4:
			category = cat_tree[1]
			train_dir, validation_dir, test_dir = get_train_test_validate_dirs(os.getcwd(), 'split_aerial')
			assign_train_test_validate(file_path, test_dir, train_dir, validation_dir)
