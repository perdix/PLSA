import numpy as np
import plsa


# Testlists
miniList = np.array( ((9,2,1,0,0,0), (8,3,2,1,0,0), (0,0,3,3,4,8), (0,2,0,2,4,7), (2,0,1,1,0,3) ))
miniTestList = np.array( ((5,5,1,0,0,0),(5,5,1,1,10,15)) )

np.set_printoptions(precision=5, suppress=True)


############################## Training #####################################################
print("\n################## Training ################### \n")
myPLSA = plsa.PLSATrain(2)
trainedTopics = myPLSA.train(miniList, 0.001)
print(trainedTopics)


############################## Testing #####################################################
print("\n################## Testing ################### \n")
myPLSA = plsa.PLSATest(2)
extractedTopics = myPLSA.query(miniTestList, 0.001)
print(extractedTopics)


