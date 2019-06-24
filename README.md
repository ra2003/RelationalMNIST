# RelationalMNIST

## How to generate the dataset:
1. set the directory for the data to go (DATA_DIR)
2. `python3 make_relational_mnist.py`

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
