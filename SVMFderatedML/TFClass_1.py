import tensorflow as tf
import csv
import numpy as np
import os
import warnings
import random
warnings.filterwarnings("ignore")
from sklearn.linear_model import SGDClassifier

class TFML:


    def dataProcess(self):
        datafolder = r'C:\Users\tingx\Downloads\MeasurementsFromAgnesDevices\Agnes-2'

        ResultFLUO25 = []
        ResultFLUO50 = []
        ResultFLUO75 = []
        ResultFLUO100 = []
        ResultVIS25 = []
        ResultVIS50 = []
        ResultVIS75 = []
        ResultVIS100 = []
        list1 = ['ResultFLUO-', 'ResultVIS-']
        list2 = [25, 50, 75, 100]
        sample2 = [2]
        #sample2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        meas1 = [x for x in range(1,21)]
        meas2 = [x for x in range(1, 41)]

        for Res in list1:
            for Portion in list2:
                tempDatalist = []
                if Portion == 100:
                    for sam in sample2:
                        for mea in meas2:
                            filenamestring = Res + str(Portion) + 'S' + str(sam) + 'M' + str(mea) + '.csv'
                            csvfile = os.path.join(datafolder, filenamestring)
                            # print(csvfile)
                            with open(csvfile, newline='') as f:
                                templist = []
                                reader = csv.reader(f)
                                data = list(reader)
                                data = data[1:]
                                # data2=[elem[2:] for elem in data]
                                for elem in data:
                                    templist.append(float(elem[2]))
                                tempDatalist.append(templist)

                else:
                    for sam in sample2:
                        for mea in meas1:
                            filenamestring = Res + str(Portion) + 'S' + str(sam) + 'M' + str(mea) + '.csv'
                            csvfile = os.path.join(datafolder, filenamestring)
                            with open(csvfile, newline='') as f:

                                templist = []
                                reader = csv.reader(f)
                                data = list(reader)
                                data = data[1:]
                                # data2=[elem[2:] for elem in data]
                                for elem in data:
                                    templist.append(float(elem[2]))
                                # if Res == 'ResultFLUO-' and Portion == 50:
                                # print(filenamestring)
                                # print(templist)
                                tempDatalist.append(templist)

                if Res == 'ResultFLUO-' and Portion == 25:
                    ResultFLUO25 = tempDatalist
                if Res == 'ResultFLUO-' and Portion == 50:
                    ResultFLUO50 = tempDatalist
                if Res == 'ResultFLUO-' and Portion == 75:
                    ResultFLUO75 = tempDatalist
                if Res == 'ResultFLUO-' and Portion == 100:
                    ResultFLUO100 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 25:
                    ResultVIS25 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 50:
                    ResultVIS50 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 75:
                    ResultVIS75 = tempDatalist
                if Res == 'ResultVIS-' and Portion == 100:
                    ResultVIS100 = tempDatalist

        splitratio = 3 / 4
        x_datalist = ResultFLUO25[int(len(ResultFLUO25) / 2):] + ResultFLUO50[int(len(ResultFLUO50) / 2):] + ResultFLUO75[int(len(ResultFLUO75) / 2):] + ResultFLUO100[int(len(ResultFLUO100) / 2):]
        y_datalist = [0] * (int(len(ResultFLUO25))-int(len(ResultFLUO25) / 2)) + [1] * (int(len(ResultFLUO50))-int(len(ResultFLUO50) / 2)) + [2] * (int(len(ResultFLUO75))-int(len(ResultFLUO75) / 2)) + [3] * (int(len(ResultFLUO100))-int(len(ResultFLUO100) / 2))

        Alldatalist = list(zip(x_datalist, y_datalist))
        random.shuffle(Alldatalist)
        x_datalist, y_datalist = zip(*Alldatalist)

        x_trainlist = x_datalist[:int(len(x_datalist) * splitratio)]
        y_trainlist = y_datalist[:int(len(y_datalist) * splitratio)]
        x_testlist = x_datalist[int(len(x_datalist) * splitratio):]
        y_testlist = y_datalist[int(len(y_datalist) * splitratio):]

        x_train = np.asarray(x_trainlist)
        y_train = np.asarray(y_trainlist)
        x_test = np.asarray(x_testlist)
        y_test = np.asarray(y_testlist)
        return x_train, y_train, x_test, y_test

    def __init__(self,name):
        self.name=name
        self.x_train,self.y_train,self.x_test, self.y_test=self.dataProcess()
        self.model = SGDClassifier(max_iter=1,tol=1e-5)#max_iter=200,


    def run_init(self):
        self.model.fit(self.x_train, self.y_train, coef_init=None, intercept_init=None)

    def run(self,coef, intercept):
        self.model.fit(self.x_train, self.y_train, coef_init=coef, intercept_init=intercept)

    def eval(self):
        print("Y_testing")
        print(self.y_train)
        print("Y Prediction")
        pred_label = self.model.predict(self.x_train)
        print(pred_label)
        errors = 0.0
        for i in range(len(pred_label)):
            if self.y_train[i] != pred_label[i]:
                errors += 1
        acc = (len(self.y_train) - errors) / len(self.y_train)
        print("Training Accuracy")
        print(acc)

        pred_label = self.model.predict(self.x_test)
        #print(pred_label)
        errors = 0.0
        for i in range(len(pred_label)):
            if self.y_test[i] != pred_label[i]:
                errors += 1
        acc = (len(self.y_test) - errors) / len(self.y_test)
        print("Test Accuracy")
        print(acc)











