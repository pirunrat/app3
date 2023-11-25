from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix

class MultinomialRegression:
    
    def __init__(self, k, n, method, alpha
                 , max_iter, regularization, l, momentum, init_method):
        
        # Momentum
        self.momentum = momentum
        
        # The regularization penalty rate
        self.l = l
        
        # Regularization
        self.regularization = regularization
        
        # k is The number of classes
        self.k = k
        
        # n is The number of features 
        self.n = n
        
        # The alpha is a learning rate
        self.alpha = alpha
        
        # The number of iterations the model should be trained
        self.max_iter = max_iter
        
        # The method of gradient updating 
        self.method = method
        
        # Initialization method
        self.init_method = init_method
    
    def fit(self, X, Y):
        #self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        
        # Initialize theta based on the chosen method
        if self.init_method == 'zeros':
            
            self.W = np.zeros((self.n, self.k))
            
        elif self.init_method == 'xavier':
            
            n = X.shape[1]
            
             # calculate the range for the weights
            lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))
             # you need to basically randomly pick weights within this range
             # generate random numbers
            numbers = np.random.rand(1000)
            scaled = lower + numbers * (upper - lower)

            # Randomly pick a number from scaled
            self.W = np.random.choice(scaled,size=(self.n,self.k))
            
            
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                
                # checking if momentum is provided
                if(self.momentum):
                    grad = self.momentum * grad + (1 -  self.momentum) * grad
                else:
                    grad = grad
                    
                    
                #check if reularization is provided
                if(self.regularization):
                    self.W = self.W - self.alpha * grad + self.regularization.derivation(self.W)
                else:
                    self.W = self.W - self.alpha * grad
                    
                
                if i % 10000 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "mini":
            start_time = time.time()
            batch_size = int(0.1 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                 
                # checking if momentum is provided
                if(self.momentum):
                    grad = self.momentum * grad + (1 -  self.momentum) * grad
                else:
                    grad = grad
                    
                    
                #check if reularization is provided
                if(self.regularization):
                    self.W = self.W - self.alpha * grad + self.regularization.derivation(self.W)
                else:
                    self.W = self.W - self.alpha * grad
                if i % 5000 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                 
                # checking if momentum is provided
                if(self.momentum):
                    grad = self.momentum * grad + (1 -  self.momentum) * grad
                else:
                    grad = grad
                    
                    
                #check if reularization is provided
                if(self.regularization):
                    self.W = self.W - self.alpha * grad + self.regularization.derivation(self.W)
                else:
                    self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 5000 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        epsilon = 1e-10 
        loss = - np.sum(Y * np.log(h + epsilon)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
       
        return self.softmax(X @ W)
    
    
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()
        plt.show()



    def confusion_plot(self, y_test,yhat):
        # Compute confusion matrix
        cm = confusion_matrix(y_test, yhat)

        # Plot using matplotlib and seaborn
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()
        
  

    def precision(self,y_true, y_pred, average, weights=None):
        
        if average == 'binary':
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))
            return true_positive / (true_positive + false_positive + 1e-9)
        
        elif average == 'macro':
            # Calculate precision for each class and take the average
            precisions = []
            for i in range(np.max(y_true) + 1):
                true_positive = np.sum((y_true == i) & (y_pred == i))
                false_positive = np.sum((y_true != i) & (y_pred == i))
                precision_i = true_positive / (true_positive + false_positive + 1e-9)
                precisions.append(precision_i)
            return np.mean(precisions)
        
        elif average == 'weighted':
            if weights is None:
                raise ValueError("Weights must be provided for 'weighted' average.")
            else:
                precision = self.recall(y_true, y_pred, average='none')
                precision = precision * weights
            return np.sum(precision)
        
        elif average == 'none':
            # Calculate precision for each class individually
            precisions = []
            for i in range(np.max(y_true) + 1):
                true_positive = np.sum((y_true == i) & (y_pred == i))
                false_positive = np.sum((y_true != i) & (y_pred == i))
                precision_i = true_positive / (true_positive + false_positive + 1e-9)
                precisions.append(precision_i)
            precisions = np.array(precisions)
            precisions[precisions == 0.0] = 1.0
            return precisions
        else:
            raise ValueError("Invalid 'average' parameter. Use 'binary', 'macro', 'weighted', or 'none'.")


    
    def recall(self,y_true, y_pred, average, weights=None):
        
        if average == 'binary':
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))
            return true_positive / (true_positive + false_negative + 1e-9)
        
        elif average == 'macro':
            # Calculate recall for each class and return a list
            recalls = []
            for i in range(np.max(y_true) + 1):
                true_positive = np.sum((y_true == i) & (y_pred == i))
                false_negative = np.sum((y_true == i) & (y_pred != i))
                recall_i = true_positive / (true_positive + false_negative + 1e-9)
                recalls.append(recall_i)
            return recalls
        
        elif average == 'weighted': 
            if weights is None:
                raise ValueError("Weights must be provided for 'weighted' average.")
            else:
                recalls = self.recall(y_true, y_pred, average='none')
                recalls = recalls * weights
            return np.sum(recalls)
        
        elif average == 'none':
            # Calculate recall for each class individually and return a list
            recalls = []
            for i in range(np.max(y_true) + 1):
                true_positive = np.sum((y_true == i) & (y_pred == i))
                false_negative = np.sum((y_true == i) & (y_pred != i))
                recall_i = true_positive / (true_positive + false_negative + 1e-9)
                recalls.append(recall_i)
            recalls = np.array(recalls)
            recalls[recalls == 0.0] = 1.0
            return recalls
        else:
            raise ValueError("Invalid 'average' parameter. Use 'binary', 'macro', 'weighted', or 'none'.")

            
            
    def f1_score(self, y_true, y_pred, average='macro', weights=None):
        precision_value = self.precision(y_true, y_pred, average)
        recall_value = self.recall(y_true, y_pred, average)

        if average == 'none':
            # Calculate F1-score for each class individually and return a list
            f1_scores = []
            for i in range(np.max(y_true) + 1):
                precision_i = precision_value[i]
                recall_i = recall_value[i]
                f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-9)
                f1_scores.append(f1_i)
            f1_scores = np.array(f1_scores)
            f1_scores[f1_scores == 0.0] = 1.0
            return f1_scores
        
        elif average == 'macro':
            # Calculate macro F1-score
            macro_precision = np.mean(precision_value)
            macro_recall = np.mean(recall_value)
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-9)
            return macro_f1
        
        elif average == 'micro':
            # Calculate micro F1-score
            micro_precision = np.sum(precision_value * y_true) / np.sum(precision_value)
            micro_recall = np.sum(recall_value * y_true) / np.sum(recall_value)
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-9)
            return micro_f1
        
        elif average == 'weighted':
            if weights is None:
                raise ValueError("Weights must be provided for 'weighted' average.")
            else:
                f1_scores = self.recall(y_true, y_pred, average='none')
                f1_scores = f1_scores * weights
            return np.sum(f1_scores)
        
        else:
            raise ValueError("Invalid 'average' parameter. Use 'none', 'macro', 'micro', or 'weighted'.")
            
    def accuracy(self,y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions


    def macro_precision(self, y_true, y_pred):
        return self.precision(y_true, y_pred, average='macro')

    def macro_recall(self, y_true, y_pred):
        return self.recall(y_true, y_pred, average='macro')

    def macro_f1_score(self, y_true, y_pred):
        return self.f1_score(y_true, y_pred, average='macro')

    def weighted_precision(self, y_true, y_pred, weights):
        return self.precision(y_true, y_pred, average='weighted', weights=weights)

    def weighted_recall(self, y_true, y_pred, weights):
        return self.recall(y_true, y_pred, average='weighted', weights=weights)

    def weighted_f1_score(self, y_true, y_pred, weights):
        return self.f1_score(y_true, y_pred, average='weighted', weights=weights)
    
    # Calculate % of each category
    def percentage_of_each_Category(self, y_train_label):
        
        series = pd.Series(y_train_label).value_counts().sort_index()
        
        return np.array(series / series.sum()).reshape(1,-1)
    
    def classificationReport(self, y_true, y_pred):
        """
        Computes the classification report, including precision, recall, f1-score, and support for each class.
        
        Parameters:
            - y_true : Array of actual class labels
            - y_pred : Array of predicted class labels
            
        Returns:
            - DataFrame : Classification report as a DataFrame
        """
        
        classes = np.unique(y_true)
        report_data = []

        for cls in classes:
            precision = self.precision(y_true, y_pred, average='binary' if len(classes) == 2 else 'none')[cls]
            recall = self.recall(y_true, y_pred, average='binary' if len(classes) == 2 else 'none')[cls]
            f1 = (2 * precision * recall) / (precision + recall + 1e-9)
            support = sum(y_true == cls)
            
            report_data.append({
                'class': cls,
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            })
        
        df = pd.DataFrame(report_data)
        df = df.set_index('class')
        
        # Adding macro and weighted averages
        avg_data = {
            'precision': np.average(df['precision'], weights=df['support']),
            'recall': np.average(df['recall'], weights=df['support']),
            'f1-score': np.average(df['f1-score'], weights=df['support']),
            'support': df['support'].sum()
        }
        
        macro_avg_data = {
            'precision': np.mean(df['precision']),
            'recall': np.mean(df['recall']),
            'f1-score': np.mean(df['f1-score']),
            'support': 'NA'
        }

        df = pd.concat([df, pd.DataFrame([avg_data], index=['weighted avg'])])
        df = pd.concat([df, pd.DataFrame([macro_avg_data], index=['macro avg'])])
        
        return df
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    

        
class Ridge(MultinomialRegression):
    
    def __init__(self, k, n, method, alpha, max_iter, regularization, l, momentum, init_method):
        self.regularization = RidgePenalty(l)
        super().__init__(k, n, method, alpha, max_iter, self.regularization, l, momentum, init_method)     