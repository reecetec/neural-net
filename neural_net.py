def dist_map(dist_id):
    dist_mapping = {
        0:"Normal",
        1:"Exponential",
        2:"Poisson",
        3:"Uniform",
        4:"Beta",
        5:"Gamma"
    }
    return dist_mapping[dist_id]

import numpy as np
class ReLu:
    def __init__(self):
        pass
    
    def func(self):
        return np.vectorize(lambda x: max(0,x))
    
    def derivative(self):
        return np.vectorize(lambda x: np.where(x > 0, 1, 0))
    
class Softmax:
    def __init__(self):
        pass
    
    def func(self):
        return lambda x: np.exp(x - np.max(x))/(np.exp(x - np.max(x)).sum())
        #return lambda x: np.exp(x)/(np.exp(x).sum())
    
    def derivative(self):
        pass
        
#defining the layers as their own classes:
class FeatureLayer:
    def __init__(self, size, act):
        self.size = size
        self.act = act #np.vectorize(lambda x: max(0,x)) #ReLu
        self.weights = None
        self.bias = None
        self.error = None
        self.neurons = np.zeros(size, dtype=np.float64)
        
class DenseLayer:
    def __init__(self, size, act):
        self.size = size
        self.act = act #np.vectorize(lambda x: max(0,x)) #ReLu
        self.weights = None
        self.bias = None
        self.error = None
        self.neurons = np.zeros(size, dtype=np.float64)

class OutputLayer:
    def __init__(self, size, act):
        self.size = size
        self.neurons = np.zeros(size, dtype=np.float64)
        self.error = None
        self.act = act #(lambda x: np.exp(x)/np.exp(x).sum()) #softmax

#categorical crossentropy assuming that softmax has already been applied
#prob = np.array([10e-100000])
#result = np.where(prob > 0, prob, -np.exp(1))
#np.log(result, out=result, where=result > 0)

def categorical_crossentropy(y_predicted,y_true):
    #get rid of log 0
    y_predicted = np.where(y_predicted > 0, y_predicted, -np.exp(1))
    return -np.sum(y_true*np.log(y_predicted,out=y_predicted, where=y_predicted > 0))/y_true.shape[0]

#Neural network class def: (for classification problems as of now)
class Neural_Net:
    def __init__(self,layers,loss_func, learning_rate):
        self.layers = layers
        self.num_layers = len(layers)
        self.sizes = [x.size for x in self.layers]
        self.init_weights()
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        
    def print_weights(self):
        print("---- Weights: ----")
        for i in range(self.num_layers-1):
            print("Layer {}:".format(i))
            print("Neurons:", self.layers[i].neurons)
            print("Weights:",self.layers[i].weights)
            print("Biases:",self.layers[i].bias)
        print("Output Neurons:",self.layers[-1].neurons)
       
    def check_nan(self):
        for i in range(self.num_layers-1):
            if np.isnan(self.layers[i].weights).any():
                print("NAN Weights Exist")
            
    def print_error(self):
        print("---- Errors: ----")
        for i in range(self.num_layers):
            print("Layer {}:".format(i))
            print(self.layers[i].error)
    
    #initializes weights
    def init_weights(self):
        for i in range(self.num_layers-1):
            self.layers[i].weights = np.array(np.random.rand(self.layers[i].size,
                                                   self.layers[i+1].size),
                                                   dtype=np.float64)
            self.layers[i].bias = np.random.rand(self.layers[i+1].size)
    
    #computes forward pass
    def forward_pass(self, feature):
        
        #input feature
        self.layers[0].neurons = feature
 
        for i in range(self.num_layers):
            #if not last layer:
            if i != self.num_layers - 1:
                #compute new neurons
                self.layers[i+1].neurons = (np.dot(self.layers[i].neurons,
                                                 self.layers[i].weights) 
                                            - self.layers[i].bias)
                #apply activation to neurons
                self.layers[i+1].neurons = self.layers[i].act.func()(self.layers[i+1].neurons)
            else:
                #just apply activation if last layer
                self.layers[i].neurons = self.layers[i].act.func()(self.layers[i].neurons)
        #print("Feature:output")
        #print(feature)
        #print(self.layers[-1].neurons)
                
    #computes backward pass
    def backward_pass(self, y_true):
        
        #get derivative of loss -> specific to categorical crossentropy / softmax:
        dldy = self.layers[-1].neurons - y_true
        dydz = self.layers[-1].neurons * (1 - self.layers[-1].neurons)

        #if first error computed in batch
        if type(self.layers[-1].error) == type(None):
            self.layers[-1].error = np.array(dldy * dydz)
        else:
            self.layers[-1].error = np.vstack([self.layers[-1].error,dldy * dydz])

        #compute error for rest of layers:
        #compute derivative then update the error for each layer
        for i in range(self.num_layers - 2,-1,-1):

            deriv = self.layers[i].act.derivative()(self.layers[i].neurons)
            error = np.dot(self.layers[i+1].error,self.layers[i].weights.T) * deriv
            
            #if first error computed in batch
            if type(self.layers[i].error) == type(None):
                self.layers[i].error = error
            else:
                #if not then append to current error matrix
                self.layers[i].error = np.vstack([self.layers[i].error,
                                                  error])
            
    #update weights after getting error from entire batch
    def update_weights(self):
        #get average error
        for i in range(self.num_layers):
            #check if 1d array: if 1d, dont take mean since this collapses vector to R1
            if self.layers[i].error.ndim != 1:
                self.layers[i].error = self.layers[i].error.mean(0)
        
        for i in range(self.num_layers - 1):
            # compute gradients
            grad_w = np.dot(self.layers[i].neurons.T, self.layers[i].error)
            grad_b = np.sum(self.layers[i].error, axis=0, keepdims=True)

            # update weights and biases
            self.layers[i].weights -= self.learning_rate * grad_w
            self.layers[i].bias -= self.learning_rate * grad_b

    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        acc_hist = []
        loss_hist = []
        avg_loss_hist = []
        for epoch in range(epochs):
            # shuffle data TODO
            #x_train, y_train = np.shuffle(x_train, y_train)
            
            # loop over batches
            for i in range(0, len(x_train), batch_size):
                # get batch
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                #update weights based on the batch
                for j in range(0,batch_size):
                    self.forward_pass(x_batch[j])
                    self.check_nan()
                    self.backward_pass(y_batch[j])
                self.update_weights()

                #track accuracy,loss
                accuracy, avg_loss, loss = self.check_accuracy(x_test,y_test)
                acc_hist.append(accuracy)
                loss_hist.append(loss)
                avg_loss_hist.append(avg_loss)
                
        return acc_hist, loss_hist, avg_loss_hist
                
                
    def predict(self,x):
        self.forward_pass(x)
        #print(self.layers[-1].neurons)
        return(self.layers[-1].neurons)
    
    
    def check_accuracy(self,x_test,y_test):
        sample_count = len(y_test)
        correct = 0
        loss = []

        for i in range(0,len(x_test)):
            model_pred = dist_map(np.argmax(self.predict(x_test[i])))
            true_val = dist_map(np.argmax(y_test[i]))
            if model_pred == true_val:
                correct += 1
            loss.append(self.loss_func(self.predict(x_test[i]),y_test[i]))
        avg_loss = np.array(loss).mean()
        return (correct/sample_count,avg_loss,loss)
    