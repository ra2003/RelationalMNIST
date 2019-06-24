# RelationalMNIST

## Requirements
- Python 3
- Numpy
- Keras (for access to MNIST and fashion MNIST data as well as image augmentation functionality)


## How to generate the dataset:
~~~
usage: generate_dataset.py [-h] [--base {mnist,fashion}] [--save_dir SAVE_DIR]
                           [--sameness {sample,class,both}]
                           [--invariants {off,on,both}]
                           [--tasks [{1,2,3,4} [{1,2,3,4} ...]]] [--fast]

Generate variations of the RelationalMNIST tasks. For more information, see
https://github.com/tannerbohn/RelationalMNIST.

optional arguments:
  -h, --help            show this help message and exit
  --base {mnist,fashion}
                        Choose what base dataset to construct the relational
                        dataset with. Either digits (MNIST) or fashion images.
  --save_dir SAVE_DIR   Specify the root folder to save the task data in.
  --sameness {sample,class,both}
                        Choose how sameness is defined for the tasks. If
                        'sample', two figures are the same only if they are
                        the same sample from the same digits or fashion class.
                        If 'class', two figures are the same if the are at
                        least from the same class.
  --invariants {off,on,both}
                        Choose whether rotation and scaling invariants are
                        added to the tasks. If 'both', multiple versions of
                        the tasks will be generated.
  --tasks [{1,2,3,4} [{1,2,3,4} ...]]
                        Choose what subset of the tasks to generate.
  --fast                Only a small number of training and test samples will
                        be generated if this argument is enabled. Use to make
                        sure things are working.
~~~

## The dataset consists of four tasks:

**Task 1:** no duplicate digits VS there are duplicates

**Task 2:** duplicates are far apart VS duplicates are near each other

**Task 3:** nearby duplicates arranged vertically VS nearby duplicates arranged horizontally

**Task 4:** both pairs of nearby duplicates have different orientation VS same orientation

For each of the four tasks, four variations can be generated, (roughly in order of difficulty):

**RelationalMNIST-S**  :  "duplicates" are exact duplicates 

**RelationalMNIST-SI** :  "duplicates" are exact duplicates, up to rotation and scaling

**RelationalMNIST-C**  :  "duplicates" are only from the same class

**RelationalMNIST-CI** :  "duplicates" are only from the same class, with rotation and scaling added

To change which of the four variation of the tasks you want, you can edit SAMENESS and INVARIANTS.
The code will currently only generate RelationalMNIST-S.


## How to load the data (example):

	with open("/home/tanner/data/RelationalMNIST-S_3.pkl", "rb") as f:
		((train_X, train_y), (test_X, test_y)) = pickle.load(f)


## Description of data:
- training and testing samples are evenly split between class 0 and 1
- train_X has 60,000 samples, test_X has 10,000 samples
- sample shape is (84, 84) -- i.e. 3X larger than MNIST digist
- samples are of type uint8 and go from 0-255, so normalizing before usage is a good idea
