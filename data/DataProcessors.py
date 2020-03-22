import logging
import pandas as pd
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split # Import train_test_split function
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
##import statsmodels.api as sm


from abc import ABC, abstractmethod, abstractproperty   ## allow for abstract base classes
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class BaseDataProcessor(ABC):
    def __init__(self, path, friendlyName):
        self.seed = int(os.environ['seed'])
        self.logger = logging.getLogger('ML7641_Logger')
        self.path = path
        self.friendlyName = friendlyName
        self.df = pd.DataFrame()
        self.trainX = pd.DataFrame()
        self.trainY = pd.DataFrame()
        self.testX = pd.DataFrame()
        self.testY = pd.DataFrame()
        self.scorer = None
        self.imagesDir = 'results/'+self.friendlyName+'/features'
        if not os.path.exists(self.imagesDir):
            os.makedirs(self.imagesDir) 

    @abstractmethod
    def load_data(self, sample_size=None):
        pass

    @abstractmethod
    def preprocess_data(self):
        ## move predictive labels to end,  drop columns, encoding, etc.
        pass

    @abstractmethod
    def transform_data(self):
        ## move predictive labels to end,  drop columns, encoding, etc.
        pass

    def apply_sampling(self, x, y):
        return (x,y)


##https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
##http://benalexkeen.com/feature-scaling-with-scikit-learn/
    def prepare_data_for_training(self, sample_size=None, test_size=0.3):
        self.load_data(sample_size)
        self.preprocess_data()
        #self.plot_datapoints()      
        self.transform_data()  
        X = self.df.iloc[:,0:-1]
        y = self.df.iloc[:,-1]
        ## scale X data 
        X = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = self.seed)   
        X_train, y_train = self.apply_sampling(X_train, y_train)   
        self.trainX = X_train		   	  			  	 		  		  		    	 		 		   		 		  
        self.trainY = y_train
        self.testX = X_test		   	  			  	 		  		  		    	 		 		   		 		  
        self.testY = y_test    
        self.X = X
        self.Y = y
        

        


    def describe_data(self):
        self.logger.info(f"Full Data Shape: {self.df.shape}")
        self.logger.info(f"X Train Shape: {self.trainX.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
        self.logger.info(f"Y Train Shape: {self.trainY.shape}")               
        self.logger.info(f"X Test Shape: {self.testX.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
        self.logger.info(f"Y Test Shape: {self.testY.shape}")
        self.logger.info(self.df.dtypes)  

    # def sns_pairplot(self):
    #     grid = sns.pairplot(self.df.iloc[:,0:-1])
    #     #plt.title("{0} Pair Grid".format(self.friendlyName))
    #     filename = self.friendlyName.replace(" ", "_")
    #     grid.savefig('{}/images/{}pairgrid.svg'.format('results', filename), format='svg', dpi=200) 
    @abstractmethod
    def plot_datapoints(self):
        pass

    def plot_categorical(self, df, features, label_col):
        cat_rows = 2
        if len(features) > 2: 
            cat_rows = math.ceil(len(features)/2)
        cat_cols = 2
        fig, axarr = plt.subplots(cat_rows,cat_cols, figsize=(20, 12))
        print(axarr.size)
        rowNum = 0
        colNum = 0
        for cat in features:
            print(cat)
            c = colNum % 2
            print(rowNum)
            print(c)
            sns.countplot(x=cat, hue = label_col,data = df, ax=axarr[rowNum][c])
            colNum = (colNum + 1)
            if colNum % 2 == 0:
                rowNum = rowNum + 1
        plt.savefig('{}/{}'.format(self.imagesDir, 'categoricalbreakdown.svg'), format='svg', dpi=200)  
        plt.close()       

    def plot_boxplot(self, df, features, label_col):
        cat_rows = 2
        if len(features) > 2: 
            cat_rows = math.ceil(len(features)/2)
        cat_cols = 2
        fig, axarr = plt.subplots(cat_rows,cat_cols, figsize=(20, 12))
        print(axarr.size)
        rowNum = 0
        colNum = 0
        for cat in features:
            print(cat)
            c = colNum % 2
            print(rowNum)
            print(c)
            sns.boxplot(y=cat, x = label_col, hue = label_col,data = df, ax=axarr[rowNum][c])
            colNum = (colNum + 1)
            if colNum % 2 == 0:
                rowNum = rowNum + 1
        plt.savefig('{}/{}'.format(self.imagesDir, 'numericalbreakdown.svg'), format='svg', dpi=200)
        plt.close()



class RedWineData(BaseDataProcessor):
    def __init__(self):
        BaseDataProcessor.__init__(self, "./data/winequality.csv","Wine Quality Data")

    def load_data(self, sample_size=None):
        self.df = pd.read_csv(self.path, header=None, names=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free sulfur','total_sulfur','density','ph','sulphates','alcohol','quality'])
        self.df = self.df.sample(frac=1, random_state=self.seed) ## mix them up
        if sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=self.seed)

    def preprocess_data(self):
        self.df['quality_class'] = 0 
        self.df.loc[self.df['quality'] > 5, 'quality_class'] = 1
        self.df.drop(columns=['quality'], inplace=True)


    def transform_data(self):
        pass    

    def plot_datapoints(self):
        label_col = 'quality_class'
        ##plot label breakdown
        df = self.df
        labels = 'Low','High'
        sizes = [df.quality_class[df[label_col]==0].count(), df.quality_class[df[label_col]==1].count()]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots(figsize=(30, 8))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        plt.title("Proportion of low quality vs high quality wine", size = 20)
        plt.savefig('{}/{}'.format(self.imagesDir, 'labelbreakdown.svg'), format='svg', dpi=200)        
        plt.close()

        sns.set_style(style="darkgrid")
        fig, axarr = plt.subplots(3,4, figsize=(50, 25), edgecolor='b', linewidth=1)
        fig.suptitle("Wine Quality Analysis", fontsize=16)
        labels = 'Low','High'
        sizes = [df.quality_class[df[label_col]==0].count(), df.quality_class[df[label_col]==1].count()]
        explode = (0, 0.1)
        axarr[0][0].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
        axarr[0][0].axis('equal')
        axarr[0][0].set_title('Wine Quality Analysis')
        sns.boxplot(y='fixed_acidity', x = label_col, hue = label_col,data = df, ax=axarr[0][1])
        sns.boxplot(y='volatile_acidity',x = label_col, hue = label_col,data = df, ax=axarr[0][2])
        sns.boxplot(y='citric_acid', x = label_col,hue = label_col,data = df, ax=axarr[0][3])
        sns.boxplot(y='residual_sugar', x = label_col,hue = label_col,data = df, ax=axarr[1][0])
        sns.boxplot(y='chlorides', x = label_col, hue = label_col,data = df, ax=axarr[1][1])
        sns.boxplot(y='free sulfur', x = label_col, hue = label_col,data = df, ax=axarr[1][2])
        sns.boxplot(y='total_sulfur', x = label_col, hue = label_col,data = df, ax=axarr[1][3])
        sns.boxplot(y='density', x = label_col, hue = label_col,data = df, ax=axarr[2][0])
        sns.boxplot(y='ph', x = label_col, hue = label_col,data = df, ax=axarr[2][1])
        sns.boxplot(y='sulphates', x = label_col, hue = label_col,data = df, ax=axarr[2][2])
        sns.boxplot(y='alcohol', x = label_col, hue = label_col,data = df, ax=axarr[2][3])

        plt.savefig('{}/{}'.format(self.imagesDir, 'dataanalysis.svg'), format='svg', dpi=200, bbox_inches = 'tight', pad_inches = 0)  

class CustomerChurnModel(BaseDataProcessor):
    def __init__(self):
        BaseDataProcessor.__init__(self, "./data/Churn_Modeling.csv","Customer Churn Data")   

    def load_data(self, sample_size=None):
        self.df = pd.read_csv(self.path)
        if sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=self.seed)        

    def preprocess_data(self):
        ## move predictive labels to end,  drop columns, encoding, etc.
        self.df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
        

    #def apply_sampling(self, x, y):
        # ##imbalanced dataset so going to use undersampling
        # temp = pd.DataFrame(x)
        # temp['Exited'] = y
        # #temp = pd.concat([x, y], axis=1)
        # exited = temp[temp.Exited==1]
        # retained = temp[temp.Exited==0]
        # downsample = resample(retained,
        #                                 replace = False, # sample without replacement
        #                                 n_samples = len(exited), # match minority n
        #                                 random_state = self.seed) # reproducible results

        # # combine minority and downsampled majority
        # temp = pd.concat([downsample, exited]).sample(frac=1)
        # X = temp.iloc[:,0:-1]
        # y = temp.iloc[:,-1]    
        # print('downsampled') 
        # return X,y 

    def transform_data(self):
        categorical_features = self.df.select_dtypes(include=['object']).columns

        #print(categorical_features[0])

        label_encode = LabelEncoder()
        #print(self.df)
        for cat in categorical_features:
            self.df[cat] = label_encode.fit_transform(self.df[cat])

        ## One Hot Encode Now
        onehotfeatures = ['HasCrCard','IsActiveMember','Exited','Geography','Gender']
        self.df = pd.get_dummies(self.df,columns=onehotfeatures)
     
        print(self.df)

    ##https://www.kaggle.com/kmalit/bank-customer-churn-prediction
    def plot_datapoints(self):
        import math
        label_col = 'Exited'
        ##plot label breakdown
        df = self.df
        labels = 'Retained','Exited'
        sizes = [df.Exited[df[label_col]==0].count(), df.Exited[df[label_col]==1].count()]
        explode = (0, 0.1)
        fig1, ax1 = plt.subplots(figsize=(30, 8))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        plt.title("Proportion of customer churned and retained", size = 20)
        plt.savefig('{}/{}'.format(self.imagesDir, 'labelbreakdown.svg'), format='svg', dpi=200)        
        plt.close()

        sns.set_style(style="darkgrid")
        fig, axarr = plt.subplots(3,4, figsize=(50, 25), edgecolor='b', linewidth=1)
        fig.suptitle("Customer Churn Data Analysis", fontsize=16)
        labels = 'Retained','Exited'
        sizes = [df.Exited[df[label_col]==0].count(), df.Exited[df[label_col]==1].count()]
        explode = (0, 0.1)
        axarr[0][0].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
        axarr[0][0].axis('equal')
        axarr[0][0].set_title('Proportion of customer churned and retained')
        sns.countplot(x='Geography', hue = label_col,data = df, ax=axarr[0][1])
        sns.countplot(x='Gender', hue = label_col,data = df, ax=axarr[0][2])
        sns.countplot(x='HasCrCard', hue = label_col,data = df, ax=axarr[0][3])
        sns.countplot(x='IsActiveMember', hue = label_col,data = df, ax=axarr[1][0])
        sns.boxplot(y='CreditScore', x = label_col, hue = label_col,data = df, ax=axarr[1][1])
        sns.boxplot(y='Age', x = label_col, hue = label_col,data = df, ax=axarr[1][2])
        sns.boxplot(y='Tenure', x = label_col, hue = label_col,data = df, ax=axarr[1][3])
        sns.boxplot(y='Balance', x = label_col, hue = label_col,data = df, ax=axarr[2][0])
        sns.boxplot(y='NumOfProducts', x = label_col, hue = label_col,data = df, ax=axarr[2][1])
        sns.boxplot(y='EstimatedSalary', x = label_col, hue = label_col,data = df, ax=axarr[2][2])
        fig.delaxes(axarr[2][3])
        plt.savefig('{}/{}'.format(self.imagesDir, 'dataanalysis.svg'), format='svg', dpi=200, bbox_inches = 'tight', pad_inches = 0)  
        # ##categorical_features = df.select_dtypes(include=['object']).columns
        # features = ['Geography', 'Gender','HasCrCard','IsActiveMember']
        # self.plot_categorical(df, features, label_col)

        # ##categorical_features = df.select_dtypes(include=['object']).columns
        # features = ['CreditScore', 'Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
        # self.plot_boxplot(df,features, label_col)
