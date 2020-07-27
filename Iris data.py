#importing the Dataset
import pandas as pd                                                #import pandas package for importing dataset
data=pd.read_csv('Iris.csv')                                       #it reads the dataset & assume the dataset in a variable.  read_csv is only read csv files
#cleaning the data
data.drop('Id',inplace=True,axis=1)                                #deleting the unwanted data in the dataset.In drop() first mention the name,inplace=True implies it is present,axis=1 implies column
data['Species'].unique()                                           #sets unique 
classes={'Iris_setosa':0,'Iris_versicolor':1,'Iris_virginica':2}   
data.replace({'Species':classes},inplace=True)                     #replace the species name as in the classes dictionary
#visualizing the data
import seaborn as sb                                               #importing seaborn package for visualization
sb.pairplot(data,hue='Species',markers=['o','s','D'])              #pairplot is visualization technique.In pairplot() first one is the name of the dataset,hue=species implies which is the target,markers=[o,s,D] implies the shape 
#Spliting the data into x & y
x=data.iloc[:,:-1].values                                          #split the data into x by avoiding last column.
y=data.iloc[:,-1].values                                           #split the data have only last column  
#spliting the data into training and testing
from sklearn.model_selection import train_test_split as tts        #importing train_test_split package for training and testing
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.25,random_state=11)  # split the x and y into testing and training
#normalizing the data
from sklearn.preprocessing import StandardScaler                   #import StandardScaler for normalizing the data
sca=StandardScaler()                                               #object created for the package
sca.fit(xtrain)                                                    #fitting the xtrain data
xtrain=sca.transform(xtrain)                                       #transforming the training data
xtest=sca.transform(xtest)       
#Algorithm Selection
#Logistic Regression
from sklearn.linear_model import LogisticRegression                #algorithm LogisticRegression is imported for the data
mlr=LogisticRegression()   
mlr.fit(xtrain,ytrain)                                             #fitting the training data
mlr.score(xtest,ytest)                                             #accuracy of the data after learning 
#KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier                 #algorithm KNeighborsClassifier is imported for the data
mknn=KNeighborsClassifier()   
mknn.fit(xtrain,ytrain) 
mknn.score(xtest,ytest)
