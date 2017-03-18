import string
import os
import numpy as np
from collections import Counter


def parse_files():

    my_path = 'C:/Users/Kyaa/Documents/GitHub/Riptide/inputdata/'

    list_of_confidence = []

    YpredTree = np.zeros((100000,))
    count = 0

    for root, dirs, file_names in os.walk(my_path):
        # print root
        # print file_names

        for f in file_names:
            count += 1
            print f
            y = np.genfromtxt(my_path + f, delimiter=",")
            # YpredTree += y[1:,1]
            list_of_confidence.append(y[1:,1])

        my_occurance_counter = 0

        for j in range(100000):  # data
            my_temp_confidence = []
            for i in range(count): # number of files
                my_temp_confidence.append(list_of_confidence[i][j])

            # print "my_length : ",len(my_temp_confidence)
            # print "temp_confidence : ", my_temp_confidence
            my_majority = Counter([round(k,1) for k in my_temp_confidence])
            value, occurance = my_majority.most_common()[0]
            # print value, occurance

            if occurance == 1:
                # print "row : ", j
                # print my_temp_confidence
                YpredTree[j] += sum(my_temp_confidence)/float(count)
                # print YpredTree[j]
            if occurance > 1:
                # my_occurance_counter += 1
                my_temp_list_inside = []
                my_temp_list_outside = []
                # print "majority : ", value, occurance
                for k in my_temp_confidence:
                    temp_difference = abs(value-k)
                    if temp_difference > 0.075:

                        my_temp_list_outside.append(k)
                    else:
                        my_temp_list_inside.append(k)

                if len(my_temp_list_inside) > 12:

                    my_occurance_counter += 1
                    value = sum(my_temp_list_inside)/float(len(my_temp_list_inside))
                    # print "mean value : ", value, ", out of ", len(my_temp_list_inside)
                else :

                    value = sum(my_temp_confidence)/float(count)
                    #print "mean value : ", value, ", out of ", len(my_temp_list_outside)

                YpredTree[j] += value

                # print my_temp_list
            else:
                # my_occurance_counter += 1       # should be 0 here
                YpredTree[j] += value

        print "my_occurance_counter : ", my_occurance_counter
        print "num of Files : ", count
        print YpredTree

        output_filename = 'results_combined/Yhat_dtree4.txt'

        # YpredTree = my_random_forest.predict_soft(x_test)

        np.savetxt(output_filename,
        np.vstack((np.arange(len(YpredTree)), YpredTree[:, ])).T,  # YpredTree[:, 1]
        '%d, %.2f', header='ID,Prob1', comments='', delimiter=',');

        print "Saved : ", output_filename

if __name__ == '__main__':
    parse_files()
