import numpy as np
import math
import csv
import sys

from entropybin import run
from entropybin.csv_processing import DATA

from scipy.stats import entropy


class DiscretizeData():
    """class responsible for managing continuous data for use in the Java SPN"""
    def __init__(self, newname, filesrc, level = 'easy',filedestination):

        #self.filepath = filepath
        #self.dataset = dataset
        self.name = newname
        self.level = level
        self.filesrc = filesrc
        self.filedestination = filedestination
        #datadirty = np.genfromtxt(filesrc + dataname, delimiter = ";")
        #remove NaN values..maybe use a boolean check instead

        #self.mean_arr = np.mean(self.dataset, axis = 0)
        #elf.sum_dataset = np.sum(self.dataset, axis = 0)
        #elf.prob_dataset = np.true_divide(self.dataset,self.sum_dataset[None, :])
        #elf.entorpy_col = entropy(self.prob_dataset)



    '''
    mean based binary binning

    creates 2 bins per feature column


    '''
    #simple discretization of continous dataset
    #converts continous data based on column means
    #split based on mean value
    def mean_split_binary_binning(self):
        d = DATA(self.filesrc)
        self.dataset = d.getdata_wholeset()
        self.mean_arr = np.mean(self.dataset, axis = 0)
        self.sum_dataset = np.sum(self.dataset, axis = 0)
        self.prob_dataset = np.true_divide(self.dataset,self.sum_dataset[None, :])

        instances, features = self.dataset.shape
        binarydata = np.zeros([instances, features])

        assert features == len(self.mean_arr)

        for row in xrange(instances):
            for feature in xrange(features):

                if self.dataset[row, feature] >= self.mean_arr[feature]:
                    binarydata[row, feature] = 1.0
                else:
                    binarydata[row, feature] = 0.0


        #binary_wine = binary_wine.astype(int)
        #need to get 75%, 15%, 10% through calculations...not ad hoc

        rows = instances
        cols = features
        train_size = int(math.floor(rows * 0.75))
        print "train_size: " + str(train_size)

        test_size = int(train_size + math.floor(rows * 0.15))
        print "test_size: " + str(test_size)

        val_size = int(test_size + math.ceil(rows*0.10))
        print "val_size: " + str(val_size)

        wine_training = binarydata[0:train_size,:].astype(int)
        wine_train_size = wine_training.shape[0]
        wine_test = binarydata[train_size:test_size,:].astype(int)
        wine_test_size = wine_test.shape[0]
        wine_valid = binarydata[test_size:,:].astype(int)
        wine_valid_size = wine_valid.shape[0]

        wine_dims_indicies = np.array([rows, cols, wine_train_size, wine_test_size, wine_valid_size ])

        np.savetxt(self.filedestination + self.name + '.dim.data', wine_dims_indicies, fmt = '%1d', delimiter = ",")


        #wine_training = binarydata[0:3672,:].astype(int)
        #wine_test = binarydata[3672:4407,:].astype(int)
        #wine_valid = binarydata[4407:4898,:].astype(int)

        #save the files
        np.savetxt(self.filedestination + self.name + '.test.data', wine_test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', wine_training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', wine_valid, fmt = '%1d', delimiter = ",")



    #simple discretization of continous dataset
    #converts continous data based on column means
    #split based on mean value
    def equal_width_bin_one_bin_per_feature(self):

        d = DATA(self.filesrc)
        self.dataset = d.getdata_wholeset()

        instances, features = self.dataset.shape

        binarydata = np.zeros([instances, features])


        split_points = np.zeros([features])
        k = 2

        for cols in xrange(features):
            mini = min(self.dataset[:,cols])
            maxi = max(self.dataset[:,cols])
            split = (maxi - mini)/float(k)
            split_point = mini + split
            split_points[cols] = split_point

        print "DD.py split_points: "
        print split_points
        for row in xrange(instances):
            for feature in xrange(features):

                if self.dataset[row, feature] >= split_points[feature]:
                    binarydata[row, feature] = 1.0
                else:
                    binarydata[row, feature] = 0.0

        y = self.dataset[:,-1]
        y_vals = set(y)

        #dorun = run.RUN()

        #big_output = dorun.buildbinaryoutput(y_vals, y)

        #binarydata = np.concatenate((binarydata[:,0:10], big_output), axis=1)

        print binarydata[0:10,:]

        #binary_wine = binary_wine.astype(int)
        #need to get 75%, 15%, 10% through calculations...not ad hoc

        rows = instances
        cols = binarydata.shape[1]
        train_size = int(math.floor(rows * 0.75))
        print "train_size: " + str(train_size)

        test_size = int(train_size + math.floor(rows * 0.15))
        print "test_size: " + str(test_size)

        val_size = int(test_size + math.ceil(rows*0.10))
        print "val_size: " + str(val_size)

        wine_training = binarydata[0:train_size,:].astype(int)
        wine_train_size = wine_training.shape[0]
        wine_test = binarydata[train_size:test_size,:].astype(int)
        wine_test_size = wine_test.shape[0]
        wine_valid = binarydata[test_size:,:].astype(int)
        wine_valid_size = wine_valid.shape[0]

        wine_dims_indicies = np.array([rows, cols, wine_train_size, wine_test_size, wine_valid_size ])

        np.savetxt(self.filedestination + self.name + '.dim.data', wine_dims_indicies, fmt = '%1d', delimiter = ",")


        #wine_training = binarydata[0:3672,:].astype(int)
        #wine_test = binarydata[3672:4407,:].astype(int)
        #wine_valid = binarydata[4407:4898,:].astype(int)

        #save the files
        np.savetxt(self.filedestination + self.name + '.test.data', wine_test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', wine_training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', wine_valid, fmt = '%1d', delimiter = ",")

    def equal_width_bin_one_bin_per_feature_output(self):

        d = DATA(self.filesrc)
        self.dataset = d.getdata_wholeset()

        instances, features = self.dataset.shape

        binarydata = np.zeros([instances, features-1])


        split_points = np.zeros([features-1])
        k = 2

        for cols in xrange(features-1):
            mini = min(self.dataset[:,cols])
            maxi = max(self.dataset[:,cols])
            split = (maxi - mini)/float(k)
            split_point = mini + split
            split_points[cols] = split_point

        print "DD.py split_points: "
        print split_points
        for row in xrange(instances):
            for feature in xrange(features -1 ):

                if self.dataset[row, feature] >= split_points[feature]:
                    binarydata[row, feature] = 1.0
                else:
                    binarydata[row, feature] = 0.0

        y = self.dataset[:,-1]
        y_vals = set(y)

        dorun = run.RUN()

        big_output = dorun.buildbinaryoutput(y_vals, y)

        binarydata = np.concatenate((binarydata, big_output), axis=1)

        print binarydata[0:10,:]

        #binary_wine = binary_wine.astype(int)
        #need to get 75%, 15%, 10% through calculations...not ad hoc

        rows = instances
        cols = binarydata.shape[1]
        train_size = int(math.floor(rows * 0.75))
        print "train_size: " + str(train_size)

        test_size = int(train_size + math.floor(rows * 0.15))
        print "test_size: " + str(test_size)

        val_size = int(test_size + math.ceil(rows*0.10))
        print "val_size: " + str(val_size)

        wine_training = binarydata[0:train_size,:].astype(int)
        wine_train_size = wine_training.shape[0]
        wine_test = binarydata[train_size:test_size,:].astype(int)
        wine_test_size = wine_test.shape[0]
        wine_valid = binarydata[test_size:,:].astype(int)
        wine_valid_size = wine_valid.shape[0]

        wine_dims_indicies = np.array([rows, cols, wine_train_size, wine_test_size, wine_valid_size ])

        np.savetxt(self.filedestination + self.name + '.dim.data', wine_dims_indicies, fmt = '%1d', delimiter = ",")


        #wine_training = binarydata[0:3672,:].astype(int)
        #wine_test = binarydata[3672:4407,:].astype(int)
        #wine_valid = binarydata[4407:4898,:].astype(int)

        #save the files
        np.savetxt(self.filedestination + self.name + '.test.data', wine_test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', wine_training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', wine_valid, fmt = '%1d', delimiter = ",")

    def equal_width_binning(self,k):
        print "logic easy equal width"

        dorun =  run.RUN(self.filesrc)
        massive_dataset = dorun.equalwidth_main(k)

        rows = massive_dataset.shape[0]
        cols = massive_dataset.shape[1]

        print "rows: " + str(rows)
        print "columns: " + str(cols)


        train_size = int(math.floor(rows * 0.75))
        print "train_size: " + str(train_size)

        test_size = int(train_size + math.floor(rows * 0.15))
        print "test_size: " + str(test_size)

        val_size = int(test_size + math.ceil(rows*0.10))
        print "val_size: " + str(val_size)

        wine_training = massive_dataset[0:train_size,:].astype(int)
        wine_train_size = wine_training.shape[0]
        wine_test = massive_dataset[train_size:test_size,:].astype(int)
        wine_test_size = wine_test.shape[0]
        wine_valid = massive_dataset[test_size:,:].astype(int)
        wine_valid_size = wine_valid.shape[0]

        wine_dims_indicies = np.array([rows, cols, wine_train_size, wine_test_size, wine_valid_size ])

        np.savetxt(self.filedestination + self.name + '.dim.data', wine_dims_indicies, fmt = '%1d', delimiter = ",")

        np.savetxt(self.filedestination + self.name + '.test.data', wine_test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', wine_training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', wine_valid, fmt = '%1d', delimiter = ",")

    def equal_width_binning_output(self,k):
        print "logic easy equal width"

        dorun =  run.RUN(self.filesrc)
        massive_dataset = dorun.equal_width_input_ouput_main(k)

        rows = massive_dataset.shape[0]
        cols = massive_dataset.shape[1]

        print "rows: " + str(rows)
        print "columns: " + str(cols)


        train_size = int(math.floor(rows * 0.75))
        print "train_size: " + str(train_size)

        test_size = int(train_size + math.floor(rows * 0.15))
        print "test_size: " + str(test_size)

        val_size = int(test_size + math.ceil(rows*0.10))
        print "val_size: " + str(val_size)

        wine_training = massive_dataset[0:train_size,:].astype(int)
        wine_train_size = wine_training.shape[0]
        wine_test = massive_dataset[train_size:test_size,:].astype(int)
        wine_test_size = wine_test.shape[0]
        wine_valid = massive_dataset[test_size:,:].astype(int)
        wine_valid_size = wine_valid.shape[0]

        wine_dims_indicies = np.array([rows, cols, wine_train_size, wine_test_size, wine_valid_size ])

        np.savetxt(self.filedestination + self.name + '.dim.data', wine_dims_indicies, fmt = '%1d', delimiter = ",")

        np.savetxt(self.filedestination + self.name + '.test.data', wine_test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', wine_training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', wine_valid, fmt = '%1d', delimiter = ",")




    def logiceasyef(self):
        print "logic easy equal frequency"


        '''
        entropy based entropy based binary division
        used entropy code where bin size is determined by random factors
        bin size is NOT fixed
	       ex.   Height <= 5.8
	             Height > 5.8

        '''
    def mdlp_binary_binning(self):

        #binary_wine = np.apply_along_axis(binary_on_mean, 1, doswine)
        dorun =  run.RUN(self.filesrc)
        massive_dataset = dorun.MDLP_main()

        rows = massive_dataset.shape[0]
        cols = massive_dataset.shape[1]
        train_size = int(math.floor(rows * 0.75))
        print "train_size: " + str(train_size)

        test_size = int(train_size + math.floor(rows * 0.15))
        print "test_size: " + str(test_size)

        val_size = int(test_size + math.ceil(rows*0.10))
        print "val_size: " + str(val_size)

        wine_training = massive_dataset[0:train_size,:].astype(int)
        wine_train_size = wine_training.shape[0]
        wine_test = massive_dataset[train_size:test_size,:].astype(int)
        wine_test_size = wine_test.shape[0]
        wine_valid = massive_dataset[test_size:,:].astype(int)
        wine_valid_size = wine_valid.shape[0]

        wine_dims_indicies = np.array([rows, cols, wine_train_size, wine_test_size, wine_valid_size ])

        np.savetxt(self.filedestination + self.name + '.dim.data', wine_dims_indicies, fmt = '%1d', delimiter = ",")

        np.savetxt(self.filedestination + self.name + '.test.data', wine_test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', wine_training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', wine_valid, fmt = '%1d', delimiter = ",")






if __name__ == "__main__":

    #pathsrc = '/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/' #will change once java starts calling this
    #read in parameters from command line d
    arguments = (sys.argv)[1:]
    #assert (len(sys.argv) - 1) == 4

    print arguments
    name = arguments[0]
    pathsrc = arguments[1]
    level = arguments[2]
    pathdest = arguments[3]

    dc = DiscretizeData(name, pathsrc,  level, pathdest)
    #dc = DiscretizeData('ewbineone','winequality-white.csv', pathsrc, pathsrc, 'binone')
    print dc.name
    print dc.filesrc
    print dc.level
    print dc.filedestination



    if dc.level == 'mean_split_binary_binning':
        #create datafiles which should be in the correct file path
        print "easy mean binning, 1 bins"
        dc.mean_split_binary_binning()


    elif dc.level == "equal_width_binning":
        print "equal width binning for moduale datasets"
        bins = arguments[4]

        if int(bins) == 1:
            print "one bins ONE ONE ONE"
            dc.equal_width_bin_one_bin_per_feature()
        else:
            print "more than one bins"
            dc.equal_width_binning(bins)

    elif dc.level == "equal_width_binning_output":
        print "equal width binning for moduale datasets with outputs"
        bins = arguments[4]
        #output_column = argument[5]

        if int(bins) == 1:
            print "one bins ONE ONE ONE"
            dc.equal_width_bin_one_bin_per_feature_output()
        else:
            print "more than one bins"
            dc.equal_width_binning_output(bins)

    elif dc.level == "equal_width_bin_wine_one":
        print"eqaul width bin one"
        dc.equal_width_bin_one_bin_per_feature_output()

    elif dc.level == 'equal_width_bin_wine':
        print "easy equal width binning 2 bins"
        bins = arguments[4]
        dc.equal_width_binning_output(bins)


    elif dc.level == 'mdlp_binary_binning':
        print "mdlp medium"
        dc.mdlp_binary_binning()


    else:
        print 'hard'
