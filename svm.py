import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import matplotlib.cm as cm
import math
import random 
import copy
import itertools

## Justin Hess
##
## Support Vector Machine
## svm.py

class SVM:
    def __init__(self, kernel="linear", sigma=5.0, visualization=True):
        # initialize our class properties
        self.kernel = kernel
        self.sigma = sigma
        self.visualization = visualization
        self.colors = {-1:'b', 1:'r'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # returns the gaussian of two vectors, or dot product in infinite dimensional space
    def gaussianKernel(self, v1, v2, sigma=None):
        if (not sigma):
            sigma = self.sigma
        #print(sigma)
        result = np.exp(-np.power(np.linalg.norm(v1-v2), 2.0) / (2 * np.power(sigma, 2.0)))
        return result
        
    # train the parameters based on the input training data
    def fit(self, X, y):

        numSamples, numFeatures = X.shape
        if (numSamples != len(y)):
            print("Error: X and y are not the same shape")

        # set the class properties to the training data

        self.n = numSamples
        # training feature data
        self.X = X
        # training data labels
        self.y = y

        # Sequential Minimal Optimization (SMO), used for training alpha, beta values with non-linearly separable data
        def SMO(X, y, C=0.05, tol=math.pow(10,-3), max_passes=50, sigma=1):
            # predict new input feature values, x, as we constructr the kernel matrix
            # this is also known as the f(x) classifier function
            def predict(X, y, a, b, x, sigma):
                result = 0.0
                for i in range(X.shape[0]):
                    # prediction result for each feature vector (row) from X with input feature vector x
                    result += (a[i]*y[i]*self.gaussianKernel(X[i,:], x, sigma))
                                
                result += b
                return result

            # each alpha, a[i], for every sample
            a = np.zeros(shape=(X.shape[0], 1))
            # initialize the bias, b
            b = 0.0
            
            # number of passes through data during training runs
            passes = 0

            # Calculate E as the error between the SVM output on the kth example and the true label, y[k], where k = (i, j)
            E = np.zeros(shape=(X.shape[0], 1))
            # create a shallow copy which will result in unwanted change of variables
            a_old = copy.deepcopy(a)

            while(passes < max_passes):
                num_changed_alphas = 0
                # for every sample
                for i in range(X.shape[0]):

                    # E is the difference between the predicted value for each feature vector X[i,:] and the true label, y[i]
                    E[i] = (predict(X,y,a,b,X[i,:],sigma) - y[i])
                
                    # verify the label times the error is greater than the tolerance and 0 < alpha[i] < C
                    if ((-y[i]*E[i] > tol and -a[i]> -C) or (y[i]*E[i] > tol and a[i] > 0)):
                        j = i

                        while (j == i):
                            # set j to any other data point other than i
                            j=random.randrange(X.shape[0]) 
            
                        # set error for other data point j
                        E[j] = (predict(X, y, a, b, X[j,:], sigma) - y[j]) 

                        # store the old alpha values for i and j
                        a_old[i] = a[i]
                        a_old[j] = a[j]
                        

                        # compute L and H values, which are the bounds that satisfy the constraint
                        if (y[i] != y[j]):
                            L=max(0, a[j] - a[i])
                            H=min(C, C + a[j] - a[i])
                        else:
                            L=max(0, a[i] + a[j] - C)
                            H=min(C, a[i] + a[j])
                
                        if (L == H):
                            continue

                        # calcualte the n parameter, using the kernel against the feature vectors of i and j
                        nu = 2*self.gaussianKernel(X[i,:], X[j,:], sigma)
                        nu = nu - self.gaussianKernel(X[i,:], X[i,:], sigma)
                        nu = nu - self.gaussianKernel(X[j,:], X[j,:], sigma)
                    
                        if (nu >= 0):
                            continue
                    
                        # clip a[j] if falls out of the bounds of L and H using E[j] and E[i] and the n parameter  
                        # solve for a[j]  
                        a[j] = a_old[j] - ((y[j]*(E[i] - E[j]))/nu)

                        if (a[j] > H):
                            a[j] = H
                        elif (a[j] < L):
                            a[j] = L
                        else:
                            # do nothing
                            pass  
            
                        if (abs(a[j] - a_old[j]) < tol):
                            continue
                    
                        # both alphas are updated
                        # solve for a[i] given we have solved for a[j]
                        a[i] += (y[i]*y[j]*(a_old[j] - a[j])) 

                        # now solve for b, now that we have optimized a[i] and a[j]

                        # calculate kernel against the feature vectors of i and j combinations
                        ii = self.gaussianKernel(X[i,:],X[i,:],sigma)
                        ij = self.gaussianKernel(X[i,:],X[j,:],sigma)
                        jj = self.gaussianKernel(X[j,:],X[j,:],sigma)			

                        # calculate b1 and b2
                        b1 = b-E[i]- (y[i]*ii*(a[i]-a_old[i]))- (y[j]*ij*(a[j]-a_old[j]))
                        b2 = b-E[j]- (y[i]*ij*(a[i]-a_old[i]))- (y[j]*jj*(a[j]-a_old[j]))

                        # calculate b from b1 and b2
                        if (a[i] > 0 and a[i] < C):
                            b = b1
                        elif (a[j] > 0 and a[j] < C):
                            b = b2
                        else:
                            b = (b1 + b2)/2.0
                    
                        num_changed_alphas += 1

                # Determine the number of alpha values that were changed this while loop iteration
                if (num_changed_alphas == 0):
                    passes += 1
                else:
                    passes = 0

            # return the alpha values (lagrange multipliers) and bias
	        return a, b      

        if (self.kernel != "linear"):
            print("Using SMO")
            # use the non-linear soft margin optimization (SMO) for training parameters
            a, b = SMO(self.X, self.y)

            print("Done with SMO")

            self.a = a
            self.b = b
            self.w = None
            return

        # dictionary where ||w|| is the key, [w, b] is the value for each iteration
        params = {}

        # create the transforms array that will hold all possible transformations of 1, -1 based on number of features
        # ex. for 2 features: [[1, 1], [-1, 1], [1, -1], [-1, -1]]
        transforms = []

        # recursive function used to find all possible combinations of 1, -1 to populate transforms array
        def ArrayRec(array, arraylist, index, length):
            arraycopy = np.copy(array)
            for j in range(0, length):
                if(j != index):
                    copy1 = np.copy(arraycopy)
                    copy2 = np.copy(arraycopy)

                    j_val = copy1[j]
                    copy1[j] = (-1)*j_val
                    copy1_a = np.copy(copy1)
                    if not any(np.array_equal(copy1_a, a_i) for a_i in arraylist):
                        arraylist.append(copy1_a)

                    copy1_b = np.copy(copy1_a)
                    ArrayRec(copy1_b, arraylist, j, length - 1)


                    j_val2 = copy2[j]
                    copy2[j] = (1)*j_val2

                    copy2_a = np.copy(copy2)
                    if not any(np.array_equal(copy1_a, a_i) for a_i in arraylist):
                        arraylist.append(copy1_a)

                    copy2_b = np.copy(copy2_a)
                    ArrayRec(copy2_b, arraylist, j, length - 1)

        # populate the transforms array
        a = np.ones((numFeatures,), dtype = int)
        transforms.append(a)
        a_neg = np.copy(a)
        a_neg = np.array([(-1)*ai for ai in a_neg])
        transforms.append(a_neg)
        for h in range(0, len(a)):
            # store a with positive head
            b1 = np.copy(a)
            b20 = np.copy(a)
            if not any(np.array_equal(b1, a_i) for a_i in transforms):
                transforms.append(b1)
            head = b1[h]
            b1[h] = (-1)*head

            b2 = np.copy(b1)
            if not any(np.array_equal(b2, a_i) for a_i in transforms):
                transforms.append(b2)

            ArrayRec(b2, transforms, h, len(a))

        # feature values
        fs = []

        for i in range(numSamples):
            xi = X[i]
            self.X[i] = X[i]
            for feature in xi:
                fs.append(feature)

        # get max and min feature values from all input training feature vectors
        self.maxFeature = max(fs)
        self.minFeature = min(fs)
        # free memory
        fs = None

        # our step sizes of iterating down weight vector to find global mimimum of data parabola
        step_sizes = [self.maxFeature * 0.1, self.maxFeature * 0.01, self.maxFeature * 0.001]

        # bias range 
        biasRange = 5
        # take smaller steps with b than w
        bMultiple = 5
        latestOptimum = self.maxFeature*10

        for step in step_sizes:
            # initialize weight vector, w, to predefined range
            # range of maximum value (multiplied by 10 previosuly) and negative of it
            w = np.array([latestOptimum, latestOptimum])

            optimized = False

            # optimized stays false until find global minimum at bottom of convex
            while not optimized:
                for b in np.arange(-1*(self.maxFeature*biasRange), 
                                        self.maxFeature*biasRange,
                                        step*bMultiple):
                    # apply each transformation to the weight vector, try to satisfy discriminant function contraint
                    for transformation in transforms:
                        # apply each transformation to the weight vector
                        w_t = w*transformation
                        found_option = True
                        # for each feature vector and label, see if satisfies general constraint
                        # yi*(xi.w+b) >= 1
                        for i in range(numSamples):
                            xi = X[i]
                            yi = y[i]
                            # check if satisfies the discriminant function general constraint
                            if not yi*(np.dot(w_t, xi)+b) >= 1:
                                found_option = False
                                break

                        # test for each transformation
                        # if this transformation of w satisfies yi(xi.w+b) >= 1
                        # we want to store the norm of it, along with its vector and b
                        if found_option:
                            # calculate norm, or magnitude of weight vector w, ||w||
                            w_mag = np.linalg.norm(w_t)
                            params[w_mag] = [w_t, b]

                # check if no combination of all transformations of weight vector and each input
                # feature vector were able to generate a hyperplane and satisfy the general constraint
                if not params:
                    print("Error! Unable to find hyperplane that satisfies constraint")
                    return

                # once finished with every b option and transformation option
                # we found the global minimum for current step size iteration
                if w[0] < 0:
                    optimized = True
                    print("Optimized w")
                # decrease the weight vector by the step size
                else:
                    w = w - step

            # sorted list of all magnitides (normalized)
            norms = sorted([n for n in params])
            # ||w|| : [w,b]               
            # ||w|| = norms[0] is smallest magnitude of weight vector
            minimized_w = norms[0]
            # get [w,b], smallest norm value vector from minimized_w key in dict
            optimizedParams = params[minimized_w]

            # ||w|| : [w, b]
            self.w = optimizedParams[0]
            self.b = optimizedParams[1]
            self.a = None

            # modify optimal value (global minimum)
            # done when close to 1
            # support vector yi(xi.w + b) = 1
            latestOptimum = optimizedParams[0][0] + step*2

    # Predict new testing values using trained parameters - classification
    def predict(self, features, labels):

        numTestFeatures = len(features)
        # our predicted labels
        y_predict = np.zeros(numTestFeatures)
        # the true labels
        y_test = labels

        y_train = self.y
        X_train = self.X
        X_test = features

        if (self.kernel != "linear"):
            # scale data
            for i in range (X_train.shape[0]):
                X_train[i,:] = [x / 100.0 for x in X_train[i,:]]

            for i in range (X_test.shape[0]):
                X_test[i,:] = [x / 100.0 for x in X_test[i,:]]

            print(X_train)

            # get our trained parameters from SMO
            a = self.a
            b = self.b

            a = a.flatten()
            print("Alphas")
            print(a)
            # Get our Support Vectors, these are the feature vectors with non-zero alphas
            sv = np.zeros(shape=(X_train.shape[0], X_train.shape[1]))
            for i in range(X_train.shape[0]):
                # threshold alphas to be at least 0.05, rather than 0.0
                if(a[i] >= 0.05):
                    sv[i] = X_train[i]

            print("Support Vectors")
            print(sv)

            self.sv = sv

            # our predicted labels
            y_predict = np.zeros(len(X_test))

            # project value but for all samples
            # do this for each vector of X, or row, should get a y_predict for each row 

            for i in range(X_test.shape[0]):
                # prediction result for this feature vector (row) from X
                result = 0
                for j in range(len(a)):
                        result += (a[j]*y_train[j]*self.gaussianKernel(X_test[i,:], sv[j]))

                y_predict[i] = result + b

            y_predict.reshape(len(X_test), 1)
            print("Predicted values")
            print(y_predict)

            # predict labels (sign of projection)
            y_predict = np.sign(y_predict)
            print("Predicted labels")
            print(y_predict)

            self.X_test = X_test
            self.y_predict = y_predict

        else:
            for i in range(numTestFeatures):
                # feature vector
                f = features[i]
                # get our trained parameters
                w = self.w
                b = self.b
                # prediction based on sign
                classification = np.sign(np.dot(np.array(f), w) + b)
                # predicted labels for test data
                y_predict[i] = classification

                if classification != 0 and self.visualization:
                    
                    self.ax.scatter(f[0], f[1], s=200, marker='*', cmap=cm.Paired, c=self.colors[classification])
                    self.ax.grid(True)
                    self.ax.xaxis.grid('grid', linestyle="--", color='black')
                    self.ax.yaxis.grid('grid', linestyle="--", color='black')

            print("Predicted labels")
            print(y_predict)

        # get the number of correct predictions, or our test accuracy
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        return y_predict

    # Plot the results and hyperplane/decision boundary
    def visualize(self):

        if (self.kernel != "linear"):
            X = self.X
            y = self.y
            grid_size = 200

            X_test = self.X_test
            y_predict = self.y_predict

            # get first two columns for plotting
            X_reshape = X[:,:2]
            X_reshape_test = X_test[:,:2]

            x_min, x_max = min(X_reshape_test[:, 0].min(), X_reshape[:, 0].min()) - 1, max(X_reshape_test[:, 0].max(), X_reshape[:, 0].max()) + 1
            y_min, y_max = min(X_reshape_test[:, 1].min(), X_reshape[:, 1].min()) - 1, max(X_reshape_test[:, 1].max(), X_reshape[:, 1].max()) + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                                np.linspace(y_min, y_max, grid_size),
                                indexing='ij')
            flatten = lambda m: np.array(m).reshape(-1,)

            def project(X, y, x):
                a = self.a
                b = self.b
                result = 0.0
                for i in range(X.shape[0]):
                    # prediction result for each feature vector (row) from X with input feature vector x
                    result += (a[i]*y[i]*self.gaussianKernel(X[i,:], x, self.sigma))

                    result += b
                    return result

            result = []
            for (i, j) in itertools.product(range(grid_size), range(grid_size)):
                point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
                result.append(project(X_reshape, y, point))

            Z = np.array(result).reshape(xx.shape)
            plt.clf()
            # plot the contour of Z, the projection of the grid points
            plt.contourf(xx, yy, Z,
                        cmap=cm.Paired,
                        levels=[-0.001, 0.001],
                        extend='both',
                        alpha=0.8)
            # train data
            plt.scatter(flatten(X_reshape[:, 0]), flatten(X_reshape[:, 1]),
                        c=flatten(y), cmap=cm.Paired)
            print(flatten(X_reshape[:, 0]))
            print(flatten(X_reshape[:, 1]))
            # test data
            plt.scatter(flatten(X_reshape_test[:, 0]), flatten(X_reshape_test[:, 1]),
                        c=flatten(y_predict), cmap=cm.Paired)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            # save figure to directory as pdf
            plt.savefig("svmnonlinear.pdf")

        else:
            for i in range(self.n):
                # plot training values used to construct hyperplane
                xi = self.X[i]
                yi = self.y[i]
                self.ax.scatter(xi[0], xi[1], s=100,cmap=cm.Paired, c=self.colors[yi])

            # add x and y axis labels
            self.ax.set_xlabel("Expression level for Gene 1")
            self.ax.set_ylabel("Expression level for Gene 2")

            # add grid
            self.ax.grid(True)
            self.ax.xaxis.grid('grid', linestyle="--", color='black')
            self.ax.yaxis.grid('grid', linestyle="--", color='black')

            # hyperplane = x.w+b
            # v = x.w+b
            def hyperplane(x,w,b,v):
                return (-w[0]*x-b+v) / w[1]

            datarange = (self.minFeature*0.9,self.maxFeature*1.1)
            hyp_x_min = datarange[0]
            hyp_x_max = datarange[1]

            # (w.x+b) = 1
            # positive support vector hyperplane
            psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
            psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
            self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

            # (w.x+b) = -1
            # negative support vector hyperplane
            nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
            nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
            self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

            # (w.x+b) = 0
            # positive support vector hyperplane
            db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
            db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
            self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

            plt.show()