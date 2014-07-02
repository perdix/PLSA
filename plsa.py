# Probabilistic Latent Semantic Analysis (PLSA) - Python implementation following Hofmann (1999)
# http://cs.brown.edu/~th/papers/Hofmann-UAI99.pdf
# Scalable using Python Memmap and parallelized NumPy methods
# 
# Paul Opitz
# 25.6.2014
# v2.1
##################################################################

from __future__ import division
import numpy as np



class PLSATrain(object): 
	
	# Public: Creates the PLSA class
	# zCount: Number of topics
	def __init__(self, zCount):
		# Setting up some class variables
		self.__trainDocs = None
		self.__kMax = zCount			# Max number of topics
		self.__iMax = None			# Max number of individual words
		self.__jMax = None			# Max number of training documents
		## Setting up training arrays
		self.__pZWDMatrix = None		# Probability of topic given word and doc (normalized)
		self.__pWZMatrix = None			# Probability of word given a topic
		self.__pZDMatrix = None			# Probability of topic given a doc


	# Public: To train the algorithm, returns the model
	# docs: Histograms of training docs, precision: Precision of the E-M iteration
	def train(self, docs, precision):
		print("Start: Training of PLSA model...") 
		# Initializing
		self.__trainDocs = docs				
		self.__iMax = docs.shape[1]		
		self.__jMax = docs.shape[0]
		print(self.__kMax)
		print(self.__iMax)
		print(self.__jMax)		
		# Using Memmap to save on disk (can be too big for memory)		
		self.__pZWDMatrix = np.memmap("pZWDMatrix.dat", dtype='float32', mode='w+', shape=(self.__kMax, self.__iMax, self.__jMax)) 
		# Random values to start	
		for i in xrange(self.__kMax):
			self.__pZWDMatrix[i] = np.random.random_sample((1,self.__iMax,self.__jMax))		
		# Creating other matrices
		self.__pWZMatrix = np.random.random_sample((self.__iMax,self.__kMax)) 				
		self.__pZDMatrix = np.random.random_sample((self.__kMax,self.__jMax)) 				
		# Training (E-M)
		self.__iterateEM(precision)
		# Save training data for later testing
		np.save("pWZMatrix.npy", self.__pWZMatrix)
		trainedTopics = self.__pZDMatrix.transpose()
		print("End: Training of PLSA model") 
		return trainedTopics

	#### Private helper methods
	
	# Do the Expectation-Maximization
	def __iterateEM(self, precision):
		oldLog = 0
		newLog = 0
		count = 0
		while (True):
			self.__updatePZWD()
			self.__updatePWZ()
			oldPZD = np.copy(self.__pZDMatrix)
			self.__updatePZD()	
			print("Iteration: " + str(count))
			#Stopping criterion
			test = np.absolute(np.subtract(oldPZD, self.__pZDMatrix))
			if np.all(test < precision):		
				break
			else:
				count += 1
				
				
	## E-Step
	def __updatePZWD(self):
		#print("E-Step")
		c = 0
		tp_pWZMatrix = np.transpose(self.__pWZMatrix)
		sumMatrix = self.__getSum(tp_pWZMatrix)
		for k in xrange(self.__kMax):		
			self.__pZWDMatrix[k] = np.outer(tp_pWZMatrix[k],self.__pZDMatrix[k]) / sumMatrix

	def __getSum(self, tp_pWZMatrix):
		sum = np.zeros((self.__iMax, self.__jMax))
		for l in xrange(self.__kMax):
			sum += np.outer(tp_pWZMatrix[l], self.__pZDMatrix[l])
		return sum

	### M-Step 1
	def __updatePWZ(self):
		#print("M-Step1")
		c = 0
		dSumMatrix = self.__getDoubleSum()
		for i in xrange(self.__iMax):
			sum = np.zeros(self.__kMax)
			for k in xrange(self.__kMax):
				col = np.transpose(self.__trainDocs)[i]
				aux = col * self.__pZWDMatrix[k][i]
				sum[k] = np.sum(aux)
			self.__pWZMatrix[i] = sum / dSumMatrix

	def __getDoubleSum(self):
		dsum = np.zeros(self.__kMax)
		transposedDocs = np.transpose(self.__trainDocs)
		for k in xrange(self.__kMax):
			aux = transposedDocs * self.__pZWDMatrix[k]
			dsum[k] = np.sum(aux)
		return dsum

	### M-Step 2
	def __updatePZD(self):
		#print("M-Step2")
		c = 0
		den = np.sum(self.__trainDocs, 1)
		for k in xrange(self.__kMax):
			aux_ji = np.transpose(self.__pZWDMatrix[k])
			sum = np.zeros(self.__jMax)
			for j in xrange(self.__jMax):
				row = self.__trainDocs[j]
				aux = row * aux_ji[j]
				sum[j] = np.sum(aux)
			self.__pZDMatrix[k] = sum / den




class PLSATest(object):

	# Public: Creates the PLSA class
	# zCount: Number of topics
	def __init__(self, zCount):
		# Setting up some class variables
		self.__testDocs = None
		self.__kMax = zCount				# Max number of topics
		self.__iMax = None				# Max number of individual words
		self.__jMaxNew = None				# Max number of testing documents
		## Setting up training arrays
		self.__pZWnewDMatrix = None			# Probability of topic given word and doc (normalized)
		## Setting up testing arrays
		self.__pWZMatrix = np.load("pWZMatrix.npy")	# Probability of topic given word and doc (normalized)
		self.__pZnewDMatrix = None			# Probability of topic given a doc



	# Public: To find suitable topics (requires a trained model)
	# docs: Histograms of testing docs, precision: Precision of the Folding-in iteration
 	def query(self, docs, precision):
		print("Start: Folding-in of PLSA model...")
		# Initializing
		self.__testDocs = docs
		self.__iMax = docs.shape[1]		
		self.__jMaxNew = docs.shape[0]
		self.__pZWnewDMatrix = np.random.random_sample((self.__kMax,self.__iMax,self.__jMaxNew)) 
		self.__pZnewDMatrix = np.random.random_sample((self.__kMax,self.__jMaxNew)) 
		# Training (Folding-in E-M)
		self.__foldingIn(precision)
		testTopics = self.__pZnewDMatrix.transpose()
		print("End: Folding-in of PLSA model")
		return testTopics


	# Do the Folding-In (Reduced E-M)
	def __foldingIn(self, precision):
		oldLog = 0
		newLog = 0
		count = 0
		while (True):
			self.__updatePZWnewD()
			oldPZnewD = np.copy(self.__pZnewDMatrix)
			self.__updatePZnewD()
			print("Iteration: " + str(count))
			#Stopping criterion	
			test = np.absolute(np.subtract(oldPZnewD, self.__pZnewDMatrix))
			if np.all(test < precision):
				break
			else:
				count += 1

	## E-Step Folding-In
	def __updatePZWnewD(self):
		#print("E-Step")
		c = 0
		tp_pWZMatrix = np.transpose(self.__pWZMatrix)
		sumMatrix = self.__getSum(tp_pWZMatrix)
		for k in xrange(self.__kMax):		
			self.__pZWnewDMatrix[k] = np.outer(tp_pWZMatrix[k],self.__pZnewDMatrix[k]) / sumMatrix

	def __getSum(self, tp_pWZMatrix):
		sum = np.zeros((self.__iMax, self.__jMaxNew))
		for l in xrange(self.__kMax):
			sum += np.outer(tp_pWZMatrix[l], self.__pZnewDMatrix[l])
		return sum	
	
	## M-Step 1 Folding-In	
	# Not necessary, kept fixed

	## M-Step 2 Folding-In	
	def __updatePZnewD(self):
		#print("M-Step2")
		c = 0
		den = np.sum(self.__testDocs, 1)
		for k in xrange(self.__kMax):
			aux_ji = np.transpose(self.__pZWnewDMatrix[k])
			sum = np.zeros(self.__jMaxNew)
			for j in xrange(self.__jMaxNew):
				row = self.__testDocs[j]
				aux = row * aux_ji[j]
				sum[j] = np.sum(aux)
			self.__pZnewDMatrix[k] = sum / den
			


