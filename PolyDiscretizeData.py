import numpy as np
import pandas as pd
import math
import csv
import sys
import argparse


from scipy.stats import entropy
from sklearn.preprocessing import OneHotEncoder

class PolyDiscretizeData():

    def __init__(self, newname, filesrc, level, filedestination, onehot):
        self.name = newname
        self.level = level
        self.filesrc = filesrc
        self.filedestination = filedestination
        self.onehot = onehot

    def basicbinning(self, feat_col):
        minimum = min(feat_col)
        maximum = max(feat_col)
        size = feat_col.shape[0]
        #print "basicbinning: size"
        #print size
        new_col = np.zeros(size)

        split = (maximum - minimum)/float(2.)
        #print split
        for rows in xrange(len(feat_col)):
                if feat_col[rows] >= split + minimum:
                    new_col[rows] = 1.
                else:
                    new_col[rows] = 0.
        #print "basicbinning: new_col.shape"
        #print new_col.shape
        return new_col.reshape((size,1))


    def getdata_wholeset(self):
        #np.random.seed(1111)
        numpcsv = np.genfromtxt(self.filesrc,delimiter=",")
        #numpcsv = numpcsv[~np.isnan(numpcsv).any(axis=1)]
        #np.random.shuffle(numpcsv)
        #print(numpcsv)
        print(numpcsv.shape)
        #instances = numpcsv.shape[0]
        #numpcsv = np.reshape(numpcsv,[instances,1])

        return numpcsv

    def equal_width_binning(self, vector, k):
        enc = OneHotEncoder()

        indecies, bins = pd.cut(vector, k, labels = False, retbins = True)
        print("indecies: ")
        print (indecies)
        print (bins)
        indecies = indecies.reshape(-1,1)
        ew = enc.fit(indecies)

        equal_width_matrix = ew.transform(indecies).toarray()
        columns = equal_width_matrix.shape[1]
        column_type = np.ones([columns])
        print ("column_types: ")
        print (column_type)

        return equal_width_matrix, column_type, bins

    def old_equal_width_binning(self, vector, k):
        rows = vector.shape[0]
        minval = min(vector)
        maxval = max(vector)
        split = (maxval - minval)/float(k)
        intervals = np.zeros(k - 1)
        for c in xrange(len(intervals)):
            #print "k: " + str(c)
            intervals[c] = ((c + 1) * split) + minval

        ew_matrix = np.zeros([rows,k])

        for inst in xrange(rows):
            val = vector[inst]
            index = 0
            for split in xrange(intervals.shape[0]):
                if val < intervals[split]:
                    ew_matrix[inst,split] = 1
                    break


                #if at the end
                if split == intervals.shape[0] - 1:
                    ew_matrix[inst,split + 1] = 1
        columns = ew_matrix.shape[1] #sjould be size k
        column_type = np.ones([columns])

        return ew_matrix, column_type, k

    def one_hot_encode(self, vector):
        enc = OneHotEncoder()
        #print "enc:"
        #print enc
        #print(vector)
        for i in xrange(vector.shape[0]):
            if vector[i] < 0:
                vector[i] = vector[i] * -10
        new_vector =  vector.reshape(-1,1)
        #print new_vector.shape
        #print("and now the vector-----")
        #print(vector)
        hot_code = enc.fit(new_vector)
        one_hot_matrix = hot_code.transform(new_vector).toarray()
        columns = one_hot_matrix.shape[1]
        column_type = np.zeros([columns])

        return one_hot_matrix, column_type



    def build_equalwidth_dataset(self, k):
        #use onehot to identify cont. and category feature columns
        dataset = self.getdata_wholeset()
        instances, features = dataset.shape

        binary_dataset = np.zeros(instances).reshape(-1,1)
        feature_types = np.zeros(1)
        old_indecies = np.zeros(1)
        feature_index = np.zeros(1)
        for v in xrange(self.onehot.shape[0]):
            if self.onehot[v] == 0:#its a one-hot encoded vector
                one_hot_feature, zero_vector = self.one_hot_encode(dataset[:, v])
                binary_dataset = np.concatenate((binary_dataset, one_hot_feature ), axis=1)
                feature_types = np.concatenate((feature_types, zero_vector),axis = 0)
                old_indecies = np.concatenate((old_indecies, np.repeat(v,one_hot_feature.shape[1])))
                feature_index = np.concatenate((feature_index, np.arange(one_hot_feature.shape[1])))
            else:
                continuous_feature, one_vector, _ = self.old_equal_width_binning(dataset[:, v], k)
                binary_dataset = np.concatenate((binary_dataset, continuous_feature), axis=1)
                feature_types = np.concatenate((feature_types, one_vector),axis = 0)
                old_indecies = np.concatenate((old_indecies, np.repeat(v,continuous_feature.shape[1])))
                feature_index = np.concatenate((feature_index, np.arange(continuous_feature.shape[1])))
        #delete the zero column
        #binary_dataset = binary_dataset[:, 1:]
        #feature_types = feature_types[1:]
        #or
        print (binary_dataset.shape)
        print (feature_types.shape)
        binary_dataset = np.delete(binary_dataset, 0, 1)
        feature_types = np.delete(feature_types,0,0)
        old_indecies = np.delete(old_indecies,0,0)
        feature_index =np.delete(feature_index,0,0)
        print (feature_index)

        return binary_dataset, feature_types, old_indecies, feature_index

    def build_equalwidth_dataset_no_onehot(self, k):
        #use onehot to identify cont. and category feature columns
        dataset = self.getdata_wholeset()
        instances, features = dataset.shape

        binary_dataset = np.zeros(instances).reshape(-1,1)
        feature_types = np.zeros(1)
        old_indecies = np.zeros(1)
        feature_index = np.zeros(1)
        for v in xrange(features):
            continuous_feature, one_vector, _ = self.old_equal_width_binning(dataset[:, v], k)
            binary_dataset = np.concatenate((binary_dataset, continuous_feature), axis=1)
            feature_types = np.concatenate((feature_types, one_vector),axis = 0)
            old_indecies = np.concatenate((old_indecies, np.repeat(v,continuous_feature.shape[1])))
            feature_index = np.concatenate((feature_index, np.arange(continuous_feature.shape[1])))
        #delete the zero column
        #binary_dataset = binary_dataset[:, 1:]
        #feature_types = feature_types[1:]
        #or
        print (binary_dataset.shape)
        print (feature_types.shape)
        binary_dataset = np.delete(binary_dataset, 0, 1)
        feature_types = np.delete(feature_types,0,0)
        old_indecies = np.delete(old_indecies,0,0)
        feature_index =np.delete(feature_index,0,0)
        print (feature_index)

        return binary_dataset, feature_types, old_indecies, feature_index

    def build_meansplit_dataset(self):
        dataset = self.getdata_wholeset()
        instances, features = dataset.shape

        binary_dataset = np.zeros(instances).reshape(-1,1)
        #print "build_meansplit_dataset: binary_dataset.shape"
        #print binary_dataset.shape
        feature_types = np.ones(1)
        old_indecies = np.ones(1)
        feature_index = np.ones(1)
        for v in xrange(features):
            continuous_feature = self.basicbinning(dataset[:, v])
            binary_dataset = np.concatenate((binary_dataset, continuous_feature), axis=1)
            feature_types = np.concatenate((feature_types, [1]),axis = 0)
            old_indecies = np.concatenate((old_indecies, np.repeat(v,continuous_feature.shape[1])))
            feature_index = np.concatenate((feature_index, np.arange(continuous_feature.shape[1])))
        #delete the zero column
        #binary_dataset = binary_dataset[:, 1:]
        #feature_types = feature_types[1:]
        #or
        print( binary_dataset.shape)
        print (feature_types.shape)
        binary_dataset = np.delete(binary_dataset, 0, 1)
        feature_types = np.delete(feature_types,0,0)
        old_indecies = np.delete(old_indecies,0,0)
        feature_index =np.delete(feature_index,0,0)
        print (feature_index)

        return binary_dataset, feature_types, old_indecies, feature_index


    def build_datafiles(self, dataset, types, oldindex, featindex):


        rows ,cols = dataset.shape


        print ("rows: " + str(rows))
        print ("columns: " + str(cols))


        training_size = int(math.floor(rows * 0.75))
        print ("train_size: " + str(training_size))

        testing_size = int(training_size + math.floor(rows * 0.15))
        print ("test_size: " + str(testing_size))

        validation_size = int(testing_size + math.ceil(rows*0.10))
        print ("val_size: " + str(validation_size))

        training = dataset[0:training_size,:].astype(int)
        train_size = training.shape[0]
        print ("training: " + str(training.shape))
        test = dataset[training_size:testing_size,:].astype(int)
        test_size = test.shape[0]
        print ("test: " + str(test.shape))
        print ("test row :")
        print (test[5,:])
        valid = dataset[testing_size:-1,:].astype(int)
        valid_size = valid.shape[0]
        print ("valid: " + str(valid.shape))

        dims_indicies = np.array([rows, cols, train_size, test_size, valid_size ])

        print ("oldindex shape: " + str(oldindex.shape))
        print ("featindex shape: " + str(featindex.shape))
        print ("oldindex: ")
        print (oldindex)
        print ("featindex: ")
        print (featindex)


        np.savetxt(self.filedestination + self.name + '.dim.data', dims_indicies, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.type.data', types, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.oldindex.data', oldindex, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.featindex.data', featindex, fmt = '%1d', delimiter = ",")

        np.savetxt(self.filedestination + self.name + '.test.data', test, fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.ts.data', training,fmt = '%1d', delimiter = ",")
        np.savetxt(self.filedestination + self.name + '.valid.data', valid, fmt = '%1d', delimiter = ",")




if __name__ == "__main__":

    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--lista",  # name on the CLI - drop the `--` for positional/required parameters
      nargs="*",  # 0 or more values expected => creates a list
      type=int,
      default=[1, 2, 3],  # default if nothing is provided
    )

    CLI.add_argument(
      "--listb",
      nargs="*",
      type=int,  # any type/callable can be used here
      default=[],
    )
    #read in parameters from command line d
    arguments = (sys.argv)[1:]
    #assert (len(sys.argv) - 1) == 4

    print (arguments)
    name = arguments[0]
    pathsrc = arguments[1]
    level = arguments[2]
    pathdest = arguments[3]
    args = CLI.parse_args(arguments[4:])
    onehot = args.lista
    onehot = np.asarray(onehot)
    flag = False
    #case for when
    print ("onehot shape")
    print (onehot.shape)
    if onehot.shape[0] <=1:
        flag = True

    #pathsrc = "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/australian-dataset.csv"
    #pathdest = "/home/andreas/Documents/MScAPR/SPN-wine/continuousdata/"
    dc = PolyDiscretizeData(name, pathsrc, level, pathdest, onehot)
    #dc = PolyDiscretizeData('testaussie',pathsrc, "equal_width_binning",pathdest, np.array([0,1,1,0,0,0,1,0,0,1,0,0,1,1,0]))
    print (dc.name)
    print (dc.filesrc)
    print (dc.level)
    print (dc.filedestination)

    if dc.level == 'mean_split_binary_binning':
        #create datafiles which should be in the correct file path
        print ("easy mean binning, 1 bins")
        dataset, types, oldindex, featindex = dc.build_meansplit_dataset()

        dc.build_datafiles(dataset,types, oldindex, featindex)


    elif dc.level == "equal_width_binning":
        print ("equal width binning for polynomial learning")
        bins = args.listb
        #bins = 2
        print("from bins =" + str(bins))
        if flag == True:
            print ("working on datasets/all continuous features")
            dataset, types, oldindex, featindex = dc.build_equalwidth_dataset_no_onehot(bins[0])
        else:
            print ("using one hot vector")
            dataset, types, oldindex, featindex = dc.build_equalwidth_dataset(bins[0])
        print (dataset.shape)
        print (types.shape)
        dc.build_datafiles(dataset,types, oldindex, featindex)

    else:
        print ("nothing was done")
