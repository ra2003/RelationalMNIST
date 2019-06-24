
import os

from keras.datasets import fashion_mnist, mnist
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import pickle
import itertools
import time
import argparse




datagen = ImageDataGenerator(fill_mode="constant", cval=0)



def view(x):
	imgplot = plt.imshow(x, cmap="Greys")
	plt.show()

def transpose_list(l):
	return list(map(list, zip(*l)))


def pre_sort_data(X, y):

	# create a dictionary where each label maps to a list
	#  of all samples with that label

	data = dict()

	labels = sorted(list(set(y)))

	for l in labels:
		data[l] = []

	for xs, ys in zip(X, y):
		data[ys].append(xs)


	return data


def compose_image(atom_data, tile_data, invariants=False):

	w, h = atom_data[0][0].shape

	layers = []
	for l in range(3):
		c1 = atom_data[tile_data[l*3][0]][tile_data[l*3][1]]
		c2 = atom_data[tile_data[l*3+1][0]][tile_data[l*3+1][1]]
		c3 = atom_data[tile_data[l*3+2][0]][tile_data[l*3+2][1]]

		if invariants:
			scale = np.random.uniform(1, 1.5)
			c1 = datagen.apply_transform(c1.reshape((h, w, 1)), {'theta':np.random.uniform(-45, 45), 'zx':scale, 'zy': scale})
			scale = np.random.uniform(1, 1.5)
			c2 = datagen.apply_transform(c2.reshape((h, w, 1)), {'theta':np.random.uniform(-45, 45), 'zx':scale, 'zy': scale})
			scale = np.random.uniform(1, 1.5)
			c3 = datagen.apply_transform(c3.reshape((h, w, 1)), {'theta':np.random.uniform(-45, 45), 'zx':scale, 'zy': scale})

		layers.append(np.concatenate([c1.reshape((h, w)), c2.reshape((h, w)), c3.reshape((h, w))], axis=-1))

	combined = np.concatenate(layers, axis=0)

	return combined

def add_horiz_dups(grid, dup_data):

	# make list of position the dups can go

	valid_locations = []

	for i_row in range(3):
		for i_col in range(3-1):
			if grid[i_row][i_col]==None and grid[i_row][i_col+1]==None:
				valid_locations.append((i_row, i_col))


	np.random.shuffle(valid_locations)


	for i_d, d in enumerate(dup_data):

		n_attempt = 0
		passed = False

		while not passed and len(valid_locations) > 1:

			i_row, i_col = valid_locations.pop()

			try:
				assert(grid[i_row][i_col] == None and grid[i_row][i_col+1] == None)

				grid[i_row][i_col] = d[0]
				grid[i_row][i_col+1] = d[1]

				passed = True
			except:
				n_attempt += 1
				print("trying again...", n_attempt)

				continue

		assert passed

	return grid




def generate_task_1(atom_X, atom_y, n, sameness="sample", invariants=False):

	# each image has 9 atomic samples in it
	# positive class: two samples are the same
	# negatice class: each sample unique

	data = pre_sort_data(atom_X, atom_y)

	class_sizes = dict()
	for k in data:
		class_sizes[k] = len(data[k])

	X = []
	y = []

	for i_sample in range(n):

		class_indices = list(range(10))
		np.random.shuffle(class_indices)
		class_indices = class_indices[:9] # we only have room for 9 tiles, but there are 10 classes

		sample_indices = [np.random.randint(0, class_sizes[i_c]) for i_c in class_indices]

		tile_data = list(zip(class_indices, sample_indices))


		#np.random.shuffle(tile_data)

		ys = i_sample % 2

		if ys == 1:

			# remove the last tile
			tile_data = tile_data[:-1]

			to_duplicate = tile_data[np.random.randint(len(tile_data))]

			if sameness == "class":
				# keep the class, but change the instance
				to_duplicate = (to_duplicate[0], np.random.randint(0, class_sizes[to_duplicate[0]]))

			tile_data.append(to_duplicate)

			np.random.shuffle(tile_data)


		# compose the layer of the combined image
		xs = compose_image(data, tile_data, invariants=invariants)

		X.append(xs)
		y.append(ys)
		
		




	return np.array(X, dtype=np.uint8), np.array(y)

def generate_task_2(atom_X, atom_y, n, sameness="sample", invariants=False):

	# each image has 9 atomic samples in it
	# positive class: duplicates close together
	# negatice class: duplicates far apart

	data = pre_sort_data(atom_X, atom_y)

	class_sizes = dict()
	for k in data:
		class_sizes[k] = len(data[k])

	X = []
	y = []

	for i_sample in range(n):

		class_indices = list(range(10))
		np.random.shuffle(class_indices)
		class_indices = class_indices[:8] # we only use 7 since there are two duplicate pairs in 9 spaces

		grid = [[None for _ in range(3)] for _ in range(3)]


		# first add the proper number of horizontal dups to grid
		i_c = class_indices.pop()
		i_s_1 = np.random.randint(0, class_sizes[i_c])
		if sameness == "sample":
			i_s_2 = i_s_1
		else:
			i_s_2 = np.random.randint(0, class_sizes[i_c])

		dup_data = [((i_c, i_s_1), (i_c, i_s_2))]


		if i_sample % 2 == 0:
			# draw the duplicates far apart

			row_0 = np.random.choice([0, 1, 2])
			col_0 = np.random.choice([0, 1, 2])

			far_coords = [(x, y) for x in range(3) for y in range(3) if abs(x-col_0)+abs(y-row_0) > 1]
			np.random.shuffle(far_coords)
			col_1, row_1 = far_coords[0]

			grid[row_0][col_0] = dup_data[0][0]
			grid[row_1][col_1] = dup_data[0][1]


		else:
			# draw the duplicates close together
			grid = add_horiz_dups(grid, dup_data)



		if np.random.random() > 0.5:
			# make duplicate pair vertical
			grid = transpose_list(grid)


		# fill the remaining spots in the grid
		for i_row in range(3):
			for i_col in range(3):
				if grid[i_row][i_col] == None:
					i_c = class_indices.pop()
					i_s = np.random.randint(0, class_sizes[i_c])
					grid[i_row][i_col] = (i_c, i_s)


		# now rotate so that adding the vertical dups is same as adding horiz dups

		#print(grid)
		#print()

		tile_data = [val for sublist in grid for val in sublist]
		xs = compose_image(data, tile_data, invariants=invariants)

		ys = 0 if i_sample % 2 == 0 else 1

		X.append(xs)
		y.append(ys)




	return np.array(X, dtype=np.uint8), np.array(y)

def generate_task_3(atom_X, atom_y, n, sameness="sample", invariants=False):

	# each image has 9 atomic samples in it
	# positive class: nearby duplicates horizontally aligned
	# negatice class: nearby duplicates vertically aligned

	data = pre_sort_data(atom_X, atom_y)

	class_sizes = dict()
	for k in data:
		class_sizes[k] = len(data[k])

	X = []
	y = []

	for i_sample in range(n):

		class_indices = list(range(10))
		np.random.shuffle(class_indices)
		class_indices = class_indices[:7] # we only use 7 since there are two duplicate pairs in 9 spaces

		grid = [[None for _ in range(3)] for _ in range(3)]


		# first add the proper number of horizontal dups to grid
		i_c = class_indices.pop()
		i_s_1 = np.random.randint(0, class_sizes[i_c])
		if sameness == "sample":
			i_s_2 = i_s_1
		else:
			i_s_2 = np.random.randint(0, class_sizes[i_c])

		dup_data = [((i_c, i_s_1), (i_c, i_s_2))]

		grid = add_horiz_dups(grid, dup_data)



		# draw the duplicates far apart

		free_coords = [(x, y) for x in range(3) for y in range(3) if grid[y][x] == None]
		np.random.shuffle(free_coords)
		col_0, row_0 = free_coords.pop()

		far_coords = [(x, y) for x, y in free_coords if abs(x-col_0)+abs(y-row_0) > 1]
		np.random.shuffle(far_coords)
		col_1, row_1 = far_coords[0]

		i_c = class_indices.pop()
		i_s_1 = np.random.randint(0, class_sizes[i_c])
		if sameness == "sample":
			i_s_2 = i_s_1
		else:
			i_s_2 = np.random.randint(0, class_sizes[i_c])

		dup_data = [((i_c, i_s_1), (i_c, i_s_2))]

		grid[row_0][col_0] = dup_data[0][0]
		grid[row_1][col_1] = dup_data[0][1]



		if i_sample % 2 == 0:
			# make duplicate pair vertical
			grid = transpose_list(grid)


		# fill the remaining spots in the grid
		for i_row in range(3):
			for i_col in range(3):
				if grid[i_row][i_col] == None:
					i_c = class_indices.pop()
					i_s = np.random.randint(0, class_sizes[i_c])
					grid[i_row][i_col] = (i_c, i_s)


		# now rotate so that adding the vertical dups is same as adding horiz dups

		#print(grid)
		#print()

		tile_data = [val for sublist in grid for val in sublist]
		xs = compose_image(data, tile_data, invariants=invariants)

		ys = 0 if i_sample % 2 == 0 else 1

		X.append(xs)
		y.append(ys)




	return np.array(X, dtype=np.uint8), np.array(y)

def generate_task_4(atom_X, atom_y, n, sameness="sample", invariants=False):

	# each image has 9 atomic samples in it
	# positive class: both collocated dups have same orientations
	# negatice class: collocated dups have different orientations

	data = pre_sort_data(atom_X, atom_y)

	class_sizes = dict()
	for k in data:
		class_sizes[k] = len(data[k])

	X = []
	y = []

	for i_sample in range(n):

		class_indices = list(range(10))
		np.random.shuffle(class_indices)
		class_indices = class_indices[:7] # we only use 7 since there are two duplicate pairs in 9 spaces

		grid = [[None for _ in range(3)] for _ in range(3)]


		# first add the proper number of horizontal dups to grid
		i_c = class_indices.pop()
		i_s_1 = np.random.randint(0, class_sizes[i_c])
		if sameness == "sample":
			i_s_2 = i_s_1
		else:
			i_s_2 = np.random.randint(0, class_sizes[i_c])

		dup_data = [((i_c, i_s_1), (i_c, i_s_2))]

		grid = add_horiz_dups(grid, dup_data)


		# get second set of dup_data
		i_c = class_indices.pop()
		i_s_1 = np.random.randint(0, class_sizes[i_c])
		if sameness == "sample":
			i_s_2 = i_s_1
		else:
			i_s_2 = np.random.randint(0, class_sizes[i_c])

		dup_data = [((i_c, i_s_1), (i_c, i_s_2))]


		if i_sample % 2 == 0:
			# both different
			grid = add_horiz_dups(transpose_list(grid), dup_data)
			
		else:
			grid = add_horiz_dups(grid, dup_data)

		if np.random.random() < 0.5:
			grid = transpose_list(grid)


		# fill the remaining spots in the grid
		for i_row in range(3):
			for i_col in range(3):
				if grid[i_row][i_col] == None:
					i_c = class_indices.pop()
					i_s = np.random.randint(0, class_sizes[i_c])
					grid[i_row][i_col] = (i_c, i_s)


		# now rotate so that adding the vertical dups is same as adding horiz dups

		#print(grid)
		#print()

		tile_data = [val for sublist in grid for val in sublist]
		xs = compose_image(data, tile_data, invariants=invariants)

		ys = 0 if i_sample % 2 == 0 else 1

		X.append(xs)
		y.append(ys)




	return np.array(X, dtype=np.uint8), np.array(y)







parser = argparse.ArgumentParser(description='Generate variations of the RelationalMNIST tasks. For more information, see \
	https://github.com/tannerbohn/RelationalMNIST.')
parser.add_argument('--base', type=str, default="mnist", choices=["mnist", "fashion"], help="Choose what base dataset to construct \
	the relational dataset with. Either digits (MNIST) or fashion images.")
parser.add_argument('--save_dir', type=str, default="./RelationalMNIST/", help="Specify the root folder to save the task data in.")
parser.add_argument('--sameness', type=str, default="sample", choices=["sample", "class", "both"], help="Choose how sameness is \
	defined for the tasks. If 'sample', two figures are the same only if they are the same sample from the same digits or fashion \
	class. If 'class', two figures are the same if the are at least from the same class.")
parser.add_argument('--invariants', type=str, default="off", choices=["off", "on", "both"], help="Choose whether rotation and scaling \
	invariants are added to the tasks. If 'both', multiple versions of the tasks will be generated.")
parser.add_argument('--tasks', nargs='*', default=[1, 2, 3, 4], type=int, choices=[1, 2, 3, 4], help="Choose what subset of the \
	tasks to generate.")
parser.add_argument('--fast', action="store_true", default=False, help="Only a small number of training and test samples will be \
	generated if this argument is enabled. Use to make sure things are working.")
parsing = parser.parse_args()



DATA_DIR = parsing.save_dir
if parsing.fast:
	N_TRAIN = 60
	N_TEST = 10
else:
	N_TRAIN = 60000
	N_TEST = 10000

if parsing.sameness == "both":
	SAMENESS = ["sample", "class"]
elif parsing.sameness == "sample":
	SAMENESS = ["sample"]
elif parsing.sameness == "class":
	SAMENESS = ["class"]


if parsing.invariants == "both":
	INVARIANTS = [True, False]
elif parsing.invariants == "off":
	INVARIANTS = [False]
elif parsing.invariants == "on":
	INVARIANTS = [True]

print("base:", parsing.base)
print("save_dir:", parsing.save_dir, "--", DATA_DIR)
print("invariants:", parsing.invariants, "--", INVARIANTS)
print("sameness:", parsing.sameness, "--", SAMENESS)
print("tasks:", parsing.tasks)

'''

np.random.seed(123)

if parsing.base == "mnist":
	((trainX, trainY), (testX, testY)) = mnist.load_data()
else:
	((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()


task_functions = [generate_task_1, generate_task_2, generate_task_3, generate_task_4]

for sameness, invariants, task_num in itertools.product(SAMENESS, INVARIANTS, [1, 2, 3, 4]):

	task_directory = "{}/RMNIST-{}{}".format(DATA_DIR.rstrip("/"), sameness[0].upper(), "I" if invariants else "")

	if not os.path.exists(task_directory):
		os.makedirs(task_directory)

	filename = "{}/task_{}.pkl".format(task_directory, task_num)

	print("Generating data for {}\n\t(Task {}, sameness = {}, invariants = {})".format(filename, task_num, sameness, invariants))

	gen = task_functions[task_num-1]

	t_start = time.time()

	np.random.seed(123)

	task_data = (gen(trainX, trainY, N_TRAIN, sameness=sameness, invariants=invariants),
		 gen(testX, testY, N_TEST, sameness=sameness, invariants=invariants))

	print("\twriting to file...")

	with open(filename, "wb") as f:
		pickle.dump(task_data, f)

	t_end = time.time()

	print("\tDone. Took {:.2f} seconds\n".format(t_end - t_start))

'''