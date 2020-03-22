import numpy as np
import pandas as pd
from time import clock
import os
import argparse
import data.DataProcessors as dp
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy import linalg
import matplotlib as mpl
os.environ['seed'] = '45604'
randomSeed = 45604
np.random.seed(randomSeed)
verbose = True

##https://github.com/kylewest520/CS-7641---Machine-Learning/blob/master/Assignment%203%20Unsupervised%20Learning/CS%207641%20HW3%20Code.ipynb


from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, make_scorer,  precision_score, mean_squared_error, plot_confusion_matrix, roc_auc_score, recall_score
from sklearn.metrics import roc_auc_score, recall_score, silhouette_score, f1_score, homogeneity_score, completeness_score
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GM
from sklearn.decomposition import PCA, FastICA as ICA, FactorAnalysis as FA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RP
##from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import learning_curve

plotsdir = 'three/plots'
csvdir = 'three/csv'

dirs = [plotsdir, csvdir]

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d) 
 

def KMeansObject(n_clusters):
    return KMeans(init='random',n_clusters=n_clusters, n_init=10,random_state=randomSeed, max_iter=300, n_jobs=-1)

def EMObject(n_components, covariance_type):
    return GM(n_components=n_components,covariance_type=covariance_type,n_init=1,warm_start=True,random_state=100)


# def plot_learning_curve(clf, X, y, title="Insert Title"):
#     train_sizes = np.append(np.linspace(0.05, 0.3, 5, endpoint=False), np.linspace(0.3, 1, 10, endpoint=True))
#     train_sizes, train_scores, test_scores, fit_times, score_times = \
#     learning_curve(clf,X, y, cv=5, train_sizes=train_sizes,verbose=False, scoring='f1',random_state=randomSeed, return_times=True)
#     print(title+" Learning Curve Completed")
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     fit_times_mean = np.mean(fit_times, axis=1)
#     fit_times_std = np.std(fit_times, axis=1)
#     score_times_mean = np.mean(score_times, axis=1); 
#     score_times_std = np.std(score_times, axis=1);  #model test/prediction times

    
#     plot_LC(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, title)
#     plot_times(train_sizes, fit_times_mean, fit_times_std, score_times_mean, score_times_std, title)
    
#     return train_sizes, train_scores_mean, fit_times_mean, score_times_mean
    
def cluster_predictions(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return pred

def simple_plot(X,y, dataset, x_label, y_label, title, saveloc):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(X, y)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(dataset+' '+ title)
    d = plotsdir+"/"+dataset
    if not os.path.exists(d):
        os.makedirs(d)     
    plt.savefig(d+"/"+saveloc)

def write_to_csv(df, dataset, name):
    d = plotsdir+"/"+dataset
    if not os.path.exists(d):
        os.makedirs(d) 
    df.to_csv(d+"/"+name+'.csv')

def run_kmeans(X,y,dataset):
    kclusters = list(np.arange(2,50))
    feature_list=['sil_score','homog_score','train_time','inertia','completeness']
    d = pd.DataFrame(0, index=np.arange(len(kclusters)), columns=feature_list, dtype='float')
    for idx, k in enumerate(kclusters):
        print(k)
        km = KMeansObject(k)
        start_time = clock()
        km.fit(X)
        labels = km.labels_
        end_time = clock()
        d['sil_score'][idx] = silhouette_score(X, labels)
        d['homog_score'][idx] = homogeneity_score(y, labels)
        d['completeness'][idx] = completeness_score(y, labels)
        d['inertia'][idx] = km.inertia_
        d['train_time'][idx] = end_time - start_time

    simple_plot(kclusters, d['sil_score'], dataset, 'Clusters','SilhouetteScore','KMeans Silhouette','kmeanssilhouette.png')
    simple_plot(kclusters, d['homog_score'], dataset, 'Clusters','Homogeneity','KMeans Homogeneity','kmeanshomogeneity.png')
    simple_plot(kclusters, d['inertia'], dataset, 'Clusters','Inertia','KMeans Inertia','kmeansinertia.png')
    simple_plot(kclusters, d['completeness'], dataset, 'Clusters','Completeness','KMeans Completeness Score','kmeanscompleteness.png')
    simple_plot(kclusters, d['train_time'], dataset, 'Clusters','Train Time','KMeans Train Time','kmeanstraintime.png')    
    write_to_csv(d,dataset,'kmeansresults')

################# EM
def run_EM(X,y,dataset,covariance_type):

    #kdist =  [2,3,4,5]
    #kdist = list(range(2,51))
    kdist = list(np.arange(2,30))
    feature_list=['sil_score','homog_score','train_time','completeness','aic_score','bic_score']
    d = pd.DataFrame(0, index=np.arange(len(kdist)), columns=feature_list, dtype='float')   

    for idx, k in enumerate(kdist):        
        start_time = clock()
        em = EMObject(n_components=k,covariance_type=covariance_type).fit(X)
        end_time = clock()
        labels = em.predict(X)
        d['sil_score'][idx] = silhouette_score(X, labels)
        d['homog_score'][idx] = homogeneity_score(y, labels)
        d['completeness'][idx] = completeness_score(y, labels)
        d['train_time'][idx] = end_time - start_time  
        d['aic_score'][idx] =  em.aic(X)
        d['bic_score'][idx] = em.bic(X)         


    simple_plot(kdist, d['sil_score'], dataset, 'Components','Silhouette Score','EM Silhouette','emsilhouette.png')
    simple_plot(kdist, d['homog_score'], dataset, 'Components','Homogeneity','EM Homogeneity','emhomogeneity.png')
    simple_plot(kdist, d['completeness'], dataset, 'Components','Completeness','EM Completeness','emcompleteness.png')
    simple_plot(kdist, d['aic_score'], dataset, 'Components','AIC','EM AIC','emAIC.png')
    simple_plot(kdist, d['bic_score'], dataset, 'Components','BIC','EM BIC','emBIC.png')
    simple_plot(kdist, d['train_time'], dataset, 'Clusters','Inertia','EM Train Time','emtraintime.png')     
    write_to_csv(d,dataset,'emresults')       

   
##https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
def em_selection(X,y,title):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            print(cv_type+": "+str(n_components))
            # Fit a Gaussian mixture with EM
            gmm = GM(n_components=n_components,covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    # plt.title('BIC score per model')
    params = best_gmm.get_params()
    plt.title(title + ' Best EM: '+params['covariance_type']+' model, '+str(params['n_components'])+' components')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.set_ylabel("BIC Score")
    spl.legend([b[0] for b in bars], cv_types)

    # # Plot the winner
    # splot = plt.subplot(2, 1, 2)
    # Y_ = clf.predict(X)
    # print(clf.covariances_.shape)
    # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
    #                                         color_iter)):
    #     #v, w = linalg.eigh(cov)
    #     if not np.any(Y_ == i):
    #         continue
    #     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    #     # Plot an ellipse to show the Gaussian component
    #     #angle = np.arctan2(w[0][1], w[0][0])
    #     #angle = 180. * angle / np.pi  # convert to degrees
    #     #v = 2. * np.sqrt(2.) * np.sqrt(v)
    #     #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #     #ell.set_clip_box(splot.bbox)
    #     #ell.set_alpha(.5)
    #     #splot.add_artist(ell)

    # plt.xticks(())
    # plt.yticks(())

    # plt.subplots_adjust(hspace=.35, bottom=.02)

    d = plotsdir+"/"+title
    if not os.path.exists(d):
        os.makedirs(d) 

    plt.savefig(d+"/EM_BestFit.png")    

def evaluate_kmeans(km, X, y, title):
    start_time = clock()
    km.fit(X)
    end_time = clock()
    training_time = end_time - start_time
    y_mode_vote = cluster_predictions(y,km.labels_)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    homogen = homogeneity_score(y, y_mode_vote)
    c_score = completeness_score(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("DataSet: "+ title)    
    print("Model Training Time (s):   "+"{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(km.n_iter_))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("Homogeneity Score: "+"{:.2f}".format(homogen)+"     Completeness Score:    "+"{:.2f}".format(c_score))
    print("*****************************************************")

def evaluate_EM(em, X, y, title):
    start_time = clock()
    em.fit(X, y)
    end_time = clock()
    training_time = end_time - start_time
    
    labels = em.predict(X)
    y_mode_vote = cluster_predictions(y,labels)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("DataSet: "+title)
    print("Model Training Time (s):   "+"{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(em.n_iter_))
    print("Log-likelihood Lower Bound: {:.2f}".format(em.lower_bound_))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.savefig(title+'EM_Confusion.png')

def run_PCA(X,y,title):
    
    pca = PCA(random_state=randomSeed).fit(X) #for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax2.set_ylabel('Eigenvalues', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("PCA Explained Variance and Eigenvalues: "+ title)
    fig.tight_layout()
    d = plotsdir+"/"+title
    if not os.path.exists(d):
        os.makedirs(d)     
    plt.savefig(d +"/PCA Explained Variance and Eigenvalues.png")

def run_FA(X,y,title):
    fa = FA(random_state=randomSeed).fit(X) #get metrics for all components
    #Get the eigenvector and eigenvalues
    ev, v = linalg.eigh(fa.components_)
    print(ev)
    evDf = pd.DataFrame(ev)
    write_to_csv(evDf,title,"fa_eigen_vector")
    
def run_ICA(X,y,title):
    
    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    ica = ICA(random_state=randomSeed, whiten=True)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: "+ title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    d = plotsdir+"/"+title
    if not os.path.exists(d):
        os.makedirs(d)     
    plt.savefig(d +"/ICA Kurtosis.png")

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

def run_RP(X,y,title):
    from itertools import product   

    dims = list(np.arange(2,(X.shape[1]-1),3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)
    for i,dim in product(range(5),dims):
        rp = RP(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()

    fig, ax1 = plt.subplots()
    ax1.plot(dims,mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims,std_recon, 'm-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("Random Components for 5 Restarts: "+ title)
    fig.tight_layout()
    d = plotsdir+"/"+title
    if not os.path.exists(d):
        os.makedirs(d)     
    plt.savefig(d +"/Random Components for 5 Restarts.png")
    
def evaluate_cluster_results(X,Y,title,km_clusters, em_components, em_type):
    km = KMeans(n_clusters=km_clusters,random_state=randomSeed,n_jobs=-1)
    evaluate_kmeans(km,X,Y, title)
    df = pd.DataFrame(km.cluster_centers_)
    df.to_csv(csvdir+"/"+title+"kMeansCenters.csv")    
    
    em = GM(n_components=em_components,covariance_type=em_type,warm_start=True,random_state=randomSeed)
    evaluate_EM(em,X,Y, title)
    df = pd.DataFrame(em.means_)
    df.to_csv(csvdir+"/"+title+"EMComponentMeans.csv")

def get_pca_data(X, components):
    return PCA(n_components=components,random_state=randomSeed).fit_transform(X)

def get_ica_data(X, components):
    return ICA(n_components=components,random_state=randomSeed).fit_transform(X)
  
def get_rp_data(X, components):
    return RP(n_components=components,random_state=randomSeed).fit_transform(X)    

def get_fa_data(X, components):
    return FA(n_components=components,random_state=randomSeed).fit_transform(X)    

def reduced_clustered_data(X, Y, title, pca_components, ica_components, rp_components, fa_components):
    pcaData = get_pca_data(X, pca_components)
    icaData = get_ica_data(X, ica_components)
    rpData = get_rp_data(X, rp_components)
    faData = get_fa_data(X, fa_components)
    
    run_kmeans(pcaData,Y,title+"PCA")
    run_kmeans(icaData,Y,title+"ICA")
    run_kmeans(rpData,Y,title+"RP")
    run_kmeans(faData,Y,title+"FA")

    run_EM(pcaData,Y,title+"PCA",'full')
    run_EM(icaData,Y,title+"ICA",'full')
    run_EM(rpData,Y,title+"RP",'full') 
    run_EM(faData,Y,title+"FA",'full')       

def run_one_neural_net(X,Y):
  
        columnNames = ["Accuracy","Precision","Recall","F1","ROC AUC","SquareError","TrainTime","TestTime"]
        dfTmp = pd.DataFrame(columns=columnNames, dtype='float')

        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.30, random_state=randomSeed)
        mlp = MLPClassifier(activation='relu',hidden_layer_sizes=10,max_iter=500, solver='adam', random_state=randomSeed, verbose=False)
        
        start = clock()
        mlp.fit(trainX, trainY)
        end = clock()
        train_time = end -start
        start = clock()
        predY = mlp.predict(testX)
        end = clock()
        pred_time = end - start

        auc = roc_auc_score(testY, predY)
        f1 = f1_score(testY,predY)
        accuracy = accuracy_score(testY,predY)
        precision = precision_score(testY,predY)
        recall = recall_score(testY,predY)
        mse = mean_squared_error(testY, predY)

        print("NN Results")
        print("*****************************************************")
        print("Model Training Time (s):   "+"{:.5f}".format(train_time))
        print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
        print("F1 Score:  "+"{:.2f}".format(f1)+"     MSE:       "+"{:.2f}".format(mse))
        print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
        print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
        print("*****************************************************")        
        return [ accuracy, precision, recall,f1,auc, mse ,train_time, pred_time]    

def plot_nn_metrics(X,original, pca, ica, rp, fa, dataset, x_label, y_label, title, saveloc):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(X, original, label='Original')
    ax.plot(X, pca, label='PCA')
    ax.plot(X, ica, label='ICA')
    ax.plot(X, rp, label='RP')
    ax.plot(X, fa, label='FA')
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(dataset+' '+ title)
    plt.legend(title="Algorithm", loc="best")
    d = plotsdir+"/"+dataset
    if not os.path.exists(d):
        os.makedirs(d)     
    plt.savefig(d+"/"+saveloc)

def add_clustered_data_to_wine_data(X, title, file):
        #wine data was 15 clusters
        km = KMeansObject(15).fit(X)
        #wine data was 9 plus full
        em = EMObject(n_components=9,covariance_type='full').fit(X)
        data = pd.DataFrame(X)
        data['KM']= km.labels_
        data['EM']= em.predict(X)
        new_columns = ['KM','EM']
        dataNew = data[new_columns]
        dataNew = pd.get_dummies(dataNew,columns=new_columns).astype('category')
        data.drop(new_columns,axis=1, inplace=True)
        df = pd.concat([data,dataNew],axis=1)
        new_X = np.array(df.values,dtype='int64')  
        #write_to_csv(df,title,file)
        return new_X



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    
    # # Adding optional argument 
    parser.add_argument("-i", "--ia", help = "Run Initial Analysis", default='y')
    parser.add_argument("-c", "--cr", help = "Run Clustering Results", default='y') 
    parser.add_argument("-r", "--rc", help = "Recreate Clustering Results", default='y') 
    parser.add_argument("-n", "--nn", help = "Run Standard Neural Net", default='y') 
    parser.add_argument("-m", "--nc", help = "Run Clustered Neural Net", default='y') 

    # # Read arguments from command line 
    args = parser.parse_args() 
    customerData = dp.CustomerChurnModel()
    customerData.prepare_data_for_training()
    redWineData = dp.RedWineData()
    redWineData.prepare_data_for_training()
    dataSets = [customerData, redWineData]

    if (args.ia == 'y'):
        ## run initial clustering analysis
        ##Run the clustering algorithms on the datasets and describe what you see.
        for data in dataSets:
            run_kmeans(data.X,data.Y,data.friendlyName)
            em_selection(data.X,data.Y,data.friendlyName)
            run_EM(data.X,data.Y,data.friendlyName,'full')


        for data in dataSets:
            print("RUN PCA "+data.friendlyName)    
            run_PCA(data.X,data.Y,data.friendlyName)            
            print("RUN ICA "+data.friendlyName)    
            run_ICA(data.X,data.Y,data.friendlyName)
            print("RUN RP "+data.friendlyName)    
            run_RP(data.X,data.Y,data.friendlyName)
            print("RUN FA "+data.friendlyName)    
            run_FA(data.X,data.Y,data.friendlyName)

    # if (args.cr == 'y'):
    # ##Customer Churn Data
    #     evaluate_cluster_results(customerData.X, customerData.Y,customerData.friendlyName,9,9,'tied')

    # # Wine Data
    #     evaluate_cluster_results(redWineData.X, redWineData.Y,redWineData.friendlyName,15,9,'full')

    # Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. 
    # Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering method. 
    # You should look at all of them, but focus on the more interesting findings in your report.
    if (args.rc == 'y'):
        #customer churn
        best_pca = 5
        best_ica = 11
        best_rp = 10
        best_fa = 7
        reduced_clustered_data(customerData.X, customerData.Y, customerData.friendlyName,best_pca, best_ica, best_rp, best_fa)
        #Wine Data
        best_pca = 3
        best_ica = 8
        best_rp = 8    
        best_fa = 3
        reduced_clustered_data(redWineData.X, redWineData.Y, redWineData.friendlyName,best_pca, best_ica, best_rp, best_fa)


##Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 
# (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) 
# and rerun your neural network learner on the newly projected data.

    if (args.nn == 'y'):
        columnNames = ["Accuracy","Precision","Recall","F1","ROC AUC","SquareError","TrainTime","TestTime"]
        df___NN = pd.DataFrame(columns=columnNames, dtype='float')
        dfPCANN = pd.DataFrame(columns=columnNames, dtype='float')
        dfICANN = pd.DataFrame(columns=columnNames, dtype='float')
        df_RPNN = pd.DataFrame(columns=columnNames, dtype='float')
        df_FANN = pd.DataFrame(columns=columnNames, dtype='float')
        nndata = redWineData
        print('Running Original Data Sets')
        result = run_one_neural_net(nndata.X, nndata.Y)
        X = range(2,11)

        for i in X:
            df___NN.loc[i-2] = result

        for i in X:
            print("Number of Components: "+str(i))
            X_pca = get_pca_data(nndata.X,i)
            X_ica = get_ica_data(nndata.X,i)
            X_rp = get_rp_data(nndata.X,i)
            X_fa = get_fa_data(nndata.X,i)
            print("PCA Components: "+str(i))
            dfPCANN.loc[i-2] =run_one_neural_net(X_pca,nndata.Y)
            print("ICA Components: "+str(i))
            dfICANN.loc[i-2] =run_one_neural_net(X_ica,nndata.Y)   
            print("RP Components: "+str(i))
            df_RPNN.loc[i-2] =run_one_neural_net(X_rp,nndata.Y)     
            print("FA Components: "+str(i))
            df_FANN.loc[i-2] =run_one_neural_net(X_fa,nndata.Y)                                    

        write_to_csv(df___NN,nndata.friendlyName+" NN","original")
        write_to_csv(dfPCANN,nndata.friendlyName+" NN","pca")
        write_to_csv(dfICANN,nndata.friendlyName+" NN","ica")
        write_to_csv(df_RPNN,nndata.friendlyName+" NN","rp")
        write_to_csv(df_FANN,nndata.friendlyName+" NN","fa")

        plot_nn_metrics(X,df___NN['Accuracy'],dfPCANN['Accuracy'],dfICANN['Accuracy'],df_RPNN['Accuracy'],df_FANN['Accuracy'],nndata.friendlyName+" NN",'Components',"Accuracy","Dimension Reduction Accuracy","accuracy")
        plot_nn_metrics(X,df___NN['TrainTime'],dfPCANN['TrainTime'],dfICANN['TrainTime'],df_RPNN['TrainTime'],df_FANN['TrainTime'],nndata.friendlyName+" NN",'Components',"Training Time","Dimension Reduction Training Time","traintime")    
        plot_nn_metrics(X,df___NN['SquareError'],dfPCANN['SquareError'],dfICANN['SquareError'],df_RPNN['SquareError'],df_FANN['SquareError'],nndata.friendlyName+" NN",'Components',"Square Error","Dimension Reduction Mean Square Error","msa")    
        plot_nn_metrics(X,df___NN['F1'],dfPCANN['F1'],dfICANN['F1'],df_RPNN['F1'],df_FANN['F1'],nndata.friendlyName+"NN",'Components',"F1 Score"," Dimension Reduction F1 Score","f1")    

    if (args.nc == 'y'):
        columnNames = ["Accuracy","Precision","Recall","F1","ROC AUC","SquareError","TrainTime","TestTime"]
        df___NN = pd.DataFrame(columns=columnNames, dtype='float')
        dfPCANN = pd.DataFrame(columns=columnNames, dtype='float')
        dfICANN = pd.DataFrame(columns=columnNames, dtype='float')
        df_RPNN = pd.DataFrame(columns=columnNames, dtype='float')
        df_FANN = pd.DataFrame(columns=columnNames, dtype='float')
        nndata = redWineData

        print('Running Original Data Sets')

        fullData = add_clustered_data_to_wine_data(nndata.X,nndata.friendlyName+"Clustered NN","original_df")
        result = run_one_neural_net(fullData, nndata.Y)
        X = range(2,11)

        for i in X:
            df___NN.loc[i-2] = result

        for i in X:
            print("Number of Components: "+str(i))
            X_pca = add_clustered_data_to_wine_data(get_pca_data(nndata.X,i),nndata.friendlyName+"Clustered NN","pca_df")
            X_ica = add_clustered_data_to_wine_data(get_ica_data(nndata.X,i),nndata.friendlyName+"Clustered NN","ica_df")
            X_rp  = add_clustered_data_to_wine_data(get_rp_data(nndata.X,i),nndata.friendlyName+"Clustered NN","rp_df")
            X_fa  = add_clustered_data_to_wine_data(get_fa_data(nndata.X,i),nndata.friendlyName+"Clustered NN","fa_df")
            print("PCA Components: "+str(i))
            dfPCANN.loc[i-2] =run_one_neural_net(X_pca,nndata.Y)
            print("ICA Components: "+str(i))
            dfICANN.loc[i-2] =run_one_neural_net(X_ica,nndata.Y)   
            print("RP Components: "+str(i))
            df_RPNN.loc[i-2] =run_one_neural_net(X_rp,nndata.Y)     
            print("FA Components: "+str(i))
            df_FANN.loc[i-2] =run_one_neural_net(X_fa,nndata.Y)                                    

        write_to_csv(df___NN,nndata.friendlyName+"Clustered NN","original")
        write_to_csv(dfPCANN,nndata.friendlyName+"Clustered NN","pca")
        write_to_csv(dfICANN,nndata.friendlyName+"Clustered NN","ica")
        write_to_csv(df_RPNN,nndata.friendlyName+"Clustered NN","rp")
        write_to_csv(df_FANN,nndata.friendlyName+"Clustered NN","fa")

        plot_nn_metrics(X,df___NN['Accuracy'],dfPCANN['Accuracy'],dfICANN['Accuracy'],df_RPNN['Accuracy'],df_FANN['Accuracy'],nndata.friendlyName+"Clustered NN",'Components',"Accuracy","Dimension Reduction Accuracy","accuracy")
        plot_nn_metrics(X,df___NN['TrainTime'],dfPCANN['TrainTime'],dfICANN['TrainTime'],df_RPNN['TrainTime'],df_FANN['TrainTime'],nndata.friendlyName+"Clustered NN",'Components',"Training Time","Dimension Reduction Training Time","traintime")    
        plot_nn_metrics(X,df___NN['SquareError'],dfPCANN['SquareError'],dfICANN['SquareError'],df_RPNN['SquareError'],df_FANN['SquareError'],nndata.friendlyName+"Clustered NN",'Components',"Square Error","Dimension Reduction Mean Square Error","msa")    
        plot_nn_metrics(X,df___NN['F1'],dfPCANN['F1'],dfICANN['F1'],df_RPNN['F1'],df_FANN['F1'],nndata.friendlyName+"Clustered NN",'Components',"F1 Score"," Dimension Reduction F1 Score","f1")    

