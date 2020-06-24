#importing the Dataset
import pandas as pd
data=pd.read_csv('Iris.csv')
#cleaning the data
data.drop('Id',inplace=True,axis=1)
data['Species'].unique()
classes={'Iris_setosa':0,'Iris_versicolor':1,'Iris_virginica':2}
data.replace({'Species':classes},inplace=True)
#visualizing the data
import seaborn as sb
sb.pairplot(data,hue='Species',markers=['o','s','D'])
#Spliting the data
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
#spliting the data into training and testing
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.25,random_state=11)
#normalizing the data
from sklearn.preprocessing import StandardScaler as sc
sca=sc()
sca.fit(xtrain)
xtrain=sca.transform(xtrain)
xtest=sca.transform(xtest)
#Algorithm Selection
#Logistic Regression
from sklearn.linear_model import LogisticRegression as lr
mlr=lr()
mlr.fit(xtrain,ytrain)
mlr.score(xtest,ytest)
#KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier as knn
mknn=knn()
mknn.fit(xtrain,ytrain)
mknn.score(xtest,ytest)
