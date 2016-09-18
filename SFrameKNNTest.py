import sframe as gl
from SFrameKNN import SFrameKNN
import numpy as np


#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#data = gl.SFrame.read_csv(url, header=False)
#data_path = '/Users/jitenamin/Documents/workspace/MahineLearningJosh/MachineLearningWithJoshGordon/iris.data'
data_path = './iris.data'
data = gl.SFrame.read_csv(data_path,header=False)
data.rename({'X5': 'target'})
(train, valid) = data.random_split(.8)

#print train.column_names()
X_train = train.select_columns(['X1','X2','X3','X4'])
y_train = train.select_columns(['target'])

X_test = valid.select_columns(['X1','X2','X3','X4'])
#X_train = select_columns
#temp_row = data[0]
#temp_row.pop('target')
#print np.array(temp_row.values())
#print data[0]
#data.remove_column('target')
#print data
my_classifier = SFrameKNN() 


my_classifier.fit(X_train,y_train)

predictions = my_classifier.predict(X_test)


count_true =0
for x in range(0,valid.num_rows()): 
    
    if(predictions[x] == valid[x]['target']):
        count_true = count_true +1
  
print round((float(count_true)/float(valid.num_rows())),4)    
    





