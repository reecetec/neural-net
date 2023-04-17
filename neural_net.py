import numpy as np
class ReLu:
    def __init__(self):
        pass
    
    def func(self):
        #return np.vectorize(lambda x: max(0,x))
        return np.vectorize(lambda x: np.where(x > 0,x, 0.01*x))
    
    def derivative(self):
        return np.vectorize(lambda x: np.where(x > 0, 1, 0.01))
    
class Softmax:
    def __init__(self):
        pass
    
    def func(self):
        #subtract off max such that it avoids overflow
        return lambda x: np.exp(x - np.max(x))/((np.exp(x - np.max(x)).sum()))
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

# assumes softmax has been applied
def categorical_crossentropy(y_predicted,y_true):
    #get rid of log 0
    y_predicted = np.where(y_predicted > 0, y_predicted, -np.exp(1))
    #print(y_true*np.log(y_predicted,out=y_predicted, where=y_predicted > 0))
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
            print("Error:",self.layers[i].error)
        print("Output Neurons:",self.layers[-1].neurons)
       
    def check_nan(self):
        for i in range(self.num_layers-1):
            if np.isnan(self.layers[i].weights).any():
                print("NAN Weights Exist")
                return
            
    def print_error(self):
        print("---- Errors: ----")
        for i in range(self.num_layers):
            print("Layer {}:".format(i))
            print(self.layers[i].error)
    
    #initializes weights
    def init_weights(self):
        
        for i in range(self.num_layers-1):
            
            #xavier initalization
            stdev = np.sqrt(2 / (self.sizes[i] + self.sizes[i+1]))
            
            #self.layers[i].weights = np.array(np.random.rand(self.layers[i].size,
            #                                       self.layers[i+1].size),
            #                                       dtype=np.float64)
            #self.layers[i].bias = np.random.rand(self.layers[i+1].size)
            self.layers[i].weights = np.array(np.random.normal(0,stdev,(self.layers[i].size,
                                                   self.layers[i+1].size)),
                                                   dtype=np.float64)
            self.layers[i].bias = np.random.normal(0,stdev,self.layers[i+1].size)
    
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
                
    #computes backward pass
    def backward_pass(self, y_true):
        
        #get derivative of loss -> specific to categorical crossentropy / softmax:
        dldy = self.layers[-1].neurons - y_true
        dydz = 1

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
            grad_w = np.outer(self.layers[i].neurons.T, self.layers[i+1].error)
            grad_b = self.layers[i+1].error
            
            # update weights and biases
            self.layers[i].weights -= (self.learning_rate * grad_w)
            self.layers[i].bias -= self.learning_rate * grad_b
            
    #shuffles data for training
    def shuffle(self,x,y):
        #combine such that shuffle preserves mapping of labels to data
        combined = np.concatenate((x,y),axis=1)
        combined = np.random.permutation(combined)
        split = np.split(combined,2,axis=1)
        return split[0],split[1]

    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        acc_hist = []
        loss_hist = []
        avg_loss_hist = []
        #append initial values
        accuracy, avg_loss, loss, conf = self.check_accuracy(x_test,y_test)
        acc_hist.append(accuracy)
        loss_hist.append(loss)
        avg_loss_hist.append(avg_loss)
        
        print("Initial Stats: accuracy:", round(accuracy,2), "avg_loss:", round(avg_loss,2))
        
        for epoch in range(epochs):
            
            # shuffle data
            x_train, y_train = self.shuffle(x_train, y_train)
            
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
            accuracy, avg_loss, loss, conf = self.check_accuracy(x_test,y_test)
            acc_hist.append(accuracy)
            loss_hist.append(loss)
            avg_loss_hist.append(avg_loss)
            
            
            print("Epoch", epoch + 1, ": accuracy:", round(accuracy,4), "avg_loss:", round(avg_loss,4))
        
        return acc_hist, loss_hist, avg_loss_hist, conf
                
                
    def predict(self,x):
        self.forward_pass(x)
        return(self.layers[-1].neurons)
    
    
    def check_accuracy(self,x_test,y_test):
        sample_count = len(y_test)
        correct = 0
        loss = []
        
        #set up confusion matrix
        conf_size = len(y_test[0])
        conf = np.zeros((conf_size,conf_size))
        
        #Get the loss, accuracy
        for i in range(0,len(x_test)):
            model_pred = np.argmax(self.predict(x_test[i]))
            true_val = np.argmax(y_test[i])
            if model_pred == true_val:
                correct += 1
            loss.append(self.loss_func(self.predict(x_test[i]),y_test[i]))
            
            #update confusion matrix (true label on rows, pred label cols)
            conf[true_val][model_pred] += 1
            
        avg_loss = np.array(loss).mean()
        
        return (correct/sample_count,avg_loss,loss,conf)
    
    
    