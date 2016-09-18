
from scipy.spatial import distance
import numpy as np
import sframe as gl

def euc(a,b):
 
    return distance.euclidean(a, b)

class SFrameKNN:
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    def compare(self,x_result,prediction):
        if(x_result['target'] == prediction['target']):
            return True
        return False
             
       
    
    def predict(self,X_test):
        predictions = gl.SArrayBuilder(dtype=str)
        for row in range(0,X_test.num_rows()):
            
            label = self.closest(X_test[row])
            predictions.append(label['target'])
        pred = predictions.close()
        return pred 
    
    def closest(self,row):
        
        np_row = np.array(row.values())
        np_x_train = np.array(self.X_train[0].values())
        
        best_dist = euc(np_row,np_x_train)
        best_index = 0
        
        #Comment Since we have copied 150 records mutiple time to create large Record
        #for i in range(1,self.X_train.num_rows()):
        for i in range(1,150):
            np_x_train = np.array(self.X_train[i].values())
            dist = euc(np_row,np_x_train)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]