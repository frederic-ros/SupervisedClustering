# -*- coding: utf-8 -*-
"""
Created on Tue May  6 05:15:54 2025

@author: frederic.ros
"""
import sys
import yaml
import os

sys.path.insert(0, "dataGeneration")
sys.path.insert(0, "concurrents/PARADIGM")
sys.path.insert(0, "ToolsCode")
from argparse import ArgumentParser, Namespace
from sklearn.preprocessing import StandardScaler
import numpy as np
from F_visu import plot_density_with_projection
from F_newidee import compute_onlyinput_patterns_with_embedding
from F_modelenewidee import ProximityPredictor
from F_mutualScan import cluster_mutual_density, visualize_clusters_2d
from F_training import CreateModel 
from F_processreal import Load_reeldata, getPCA
from F_mainconcurrents import Testcompetitors, getnoisemethode, getSilhouettebase
from F_synthetic import SyntheticSet
from F_util import G_simplestatistics, compute_stats_by_criterion, compute_custom_stats, processstat, processstat_global
from F_launchdeeplearning import launchdeeplearning
from F_newidee import genpoint
from F_yaml import load_yaml_if_exists
from F_savesyntheticdata import save_xy_to_txt
import matplotlib
import matplotlib.pyplot as plt

#...............DEFINE CONSTANT......................................................
M_KMEANS = 0
M_XMEANS = 1
M_MEANSHIFT = 2
M_DBSCAN = 3
M_ADBSCAN = 4
M_DPCA = 5
M_DPCCE = 6
M_HDBSCAN = 7
M_SNN = 8
M_SPECTRAL = 9
M_RNNDBSCAN = 10
M_ANDCLUST = 11
M_POCS = 12
M_DEMUNE = 13
M_OURS = 14
M_DECSIMPLIFIED = 15

#...............DEFAULT CONFIG......................................................
def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--genpoint_s", type=bool, default=False)
    parser.add_argument("--createmodel", type=bool, default=False)
    parser.add_argument("--use_fuzzy", type=bool, default=True)
    parser.add_argument("--testmodel_synthetic", type=bool, default=False)
    parser.add_argument("--testmodel_real", type=bool, default=False)
    parser.add_argument("--deeplearning_methods", type=bool, default=False)
    parser.add_argument("--experiments_synthetic", type=bool, default=True)
    parser.add_argument("--experiments_real", type=bool, default=False)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--dev_training", type=float, default=0.3)
    parser.add_argument("--dimtraining_start", type=int, default=2)
    parser.add_argument("--dimtraining_end", type=int, default= 33)
    parser.add_argument("--ourmethod_parameter", type=float, nargs='+', default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--Hamming_ratio_training", type=float, default=0.01)   
    parser.add_argument("--n_set_per_dimension", type=int, default=1)
    parser.add_argument("--maxdev_test", type=float, default=0.33)
    parser.add_argument("--option_test_synthetique", type=int, default=0)
    parser.add_argument("--dimension_test", type=int, default=2)
    parser.add_argument("--n_synthetic_tests", type=int, default=10)
    parser.add_argument("--savesyntheticoption", type=bool, default=True)
    parser.add_argument("--savesynthethicdata", type=str, default="savesyntheticdata/level2")
    parser.add_argument("--Hamming_ratio_test", type=float, default=0.0)
    parser.add_argument("--percentage_noise", type=float, default=0.15)
    parser.add_argument("--low_method_number", type=int, default=14)
    parser.add_argument("--Up_method_number", type=int, default=15)
    #parser.add_argument("--realdata", type=str,default="Highrealdata/d32")
    parser.add_argument("--realdata", type=str,default="data2DFORME")
    #parser.add_argument("--saverealresult", type=str,default="resultsreal/mediumdim")
    parser.add_argument("--saverealresult", type=str,default="resultsreal/highdim")
    parser.add_argument("--Optionsaveresultreal", type=bool, default=True)
    parser.add_argument("--savesyntheticresult", type=str,default="resultsynthetique/all/result")
    parser.add_argument("--n_samples_per_datasets", type=int, default=1000)
    parser.add_argument("--dimension_of_p_embedding", type=int, default=10)
    parser.add_argument("--name_training_data", type=str, default="data.txt")
    #parser.add_argument("--name_model", type=str, default="models/modelhybrid100novel")
    parser.add_argument("--name_model", type=str, default="models/modelhybrid2-32novel")
    #parser.add_argument("--name_model", type=str, default="models/modelhybridbis2")
    #parser.add_argument("--name_model", type=str, default="models/modelaffichage2-32novel")
    parser.add_argument("--Option_model", type=int, default=0)
    parser.add_argument("--draw_graphics", type=bool, default=True)   

    
    return parser.parse_args()

#...............YAML CONFIG......................................................


def get_config_yaml(yaml_path: str = "config.yaml") -> Namespace:
    args = get_config()
    args = load_yaml_if_exists(yaml_path, args)
    return args

#.....................................................................
def DisplayMethod(run):
    if run[0] == 1: print("M_KMEANS")
    if run[1] == 1: print("M_XMEANS") 
    if run[2] == 1: print("M_MEANSHIFT")
    if run[3] == 1: print("M_DBSCAN") 
    if run[4] == 1: print("M_ADBSCAN")
    if run[5] == 1: print("M_DPCA") 
    if run[6] == 1: print("M_DPCCE")
    if run[7] == 1: print("M_HDBSCAN")
    if run[8] == 1: print("M_SNN")
    if run[9] == 1: print("M_SPECTRAL")
    if run[10] == 1: print("M_RNNDBSCAN")
    if run[11] == 1: print("M_ANDCLUST")
    if run[12] == 1: print("M_POCS")
    if run[13] == 1: print("M_DEMUNE")
    if run[14] == 1: print("M_OURS")
    if run[15] == 1: print("M_DECSIMPLIFIED")
    if run[16] == 1: print("M_VADE")



def getname(value=0):
    if (value==M_KMEANS): return "KMEANS"
    if (value==M_XMEANS): return "XMEANS"
    if (value==M_MEANSHIFT): return "MEANSHIFT"
    if (value==M_DBSCAN): return "DBSCAN"
    if (value==M_ADBSCAN): return "ADBSCAN"
    if (value==M_DPCA): return "RDPCA"
    if (value==M_DPCCE): return "RDPCE"
    if (value==M_HDBSCAN): return "HDBSCAN"
    if (value==M_SNN): return "SNN"
    if (value==M_SPECTRAL): return "SPECTRAL"
    if (value==M_RNNDBSCAN): return "RNNDBSCAN"
    if (value==M_ANDCLUST): return "ANDCLUST"
    if (value==M_POCS): return "POCS"
    if (value==M_DEMUNE): return "DEMUNE"
    if (value==M_OURS): return "OURS"
    if (value==M_DECSIMPLIFIED): return "DECSIMPLIFIED"
    
    return "EMPTY"



#######################################################################################   
def TestModelReal(directory_path = "realdata",p_embedding=10,
                  namemodel="model",n_samples = 1000,T_method_low=0, T_method_up=1,
                  ourmethod_parameter=[0.4,0.8],
                  nameresult="highdim/", optionPCA=False, Optionsaveresultreal = False, draw = False):
        
    draw = draw
    scaler = StandardScaler()
    save= Optionsaveresultreal
    
    run = np.zeros(20,int)
    noise_methode = np.zeros(20,int)
    getnoisemethode(noise_methode)
   
    for i in range(T_method_low, T_method_up):
        run[i] = 1
    
    noise = 0
    Competiteurs = True
      
    files, data_l, y_l = Load_reeldata(directory_path = directory_path,t = 1, start = 0)
    R=[]
    CVI=[]
    for i in range(0,len(y_l)):
       
        if data_l[i].shape[1] <= 32:
            X = data_l[i]
            X_scaled = scaler.fit_transform(X)
        else : 
            if optionPCA==True: X = getPCA(data_l[i]) #OPTION PCA.
            X = data_l[i]
            X_scaled = scaler.fit_transform(X)
        X = X_scaled #on scale!    
       
        S = getSilhouettebase(X, y_l[i],noise = 0, label_noise = -1)    
        CVI.append(S)
        print(files[i],"SHAPE",np.array(data_l[i]).shape,"CVI=",S)
        name="2D Visualization" + "(D-space="  + str(X.shape[1]) + ",CVI=" + f"{S:.3f}" +")"  
        if draw== True: visualize_clusters_2d(X, y_l[i],method='tsne', name = name)
        if Competiteurs == True: R.append(Testcompetitors(X, y_l[i], run, noise, 
                                                          noise_methode, 
                                                          namemodel=namemodel+".pth",
                                                          ourmethod_parameter=ourmethod_parameter,
                                                          draw=draw))
        
    if  (T_method_low == T_method_up - 1) and len(y_l)>=5:   #on teste une seule methode.  
        means = processstat(R,block_size=5)
        print("Means = ",means)
        means, stds,_=processstat_global(R, tri=True, CVI=CVI, threshold=0.1)
        print("Stat = ",means, stds)
        stats = np.vstack((means, stds))  # 2 lignes : mean / std
        if save==True:
            r_f =  nameresult + "methodestat"+str(T_method_low) + ".txt"
            np.savetxt(fname= r_f, X = stats,delimiter="\t",fmt="%.3f")
            r_f =  nameresult + "methode"+str(T_method_low) + ".txt"  
            np.savetxt(fname= r_f, X = np.array(R).squeeze(),delimiter="\t",fmt="%.3f")    
    
    
#######################################################################################   
def TestModelSynthetique(n=1,dim=2, max_dev = 0.4,hamming_test=0.1,
                         p_embedding=10,option=0,
                         ourmethod_parameter=[0.4,0,8],
                         n_samples = 1000,
                         p_noise=0.05,T_method_low=0, T_method_up=1,
                         Saveresult = "resultsynthetique/all/result",
                         savesynthethicdata = "savesyntheticdata/level1",
                         savesyntheticoption = False,
                         namemodel="model", draw=True):
    
                            
    Kv_in = 16
    Kv_out = 16
    
    Competiteurs = True
    Our = False
    max_dev = max_dev
     
    run = np.zeros(20,int)
    noise_methode = np.zeros(20,int)
    getnoisemethode(noise_methode)
   

    for i in range(T_method_low, T_method_up):
        run[i] = 1
        
    noise = 1
    
    input_dim = Kv_in * Kv_in + p_embedding 
    output_dim = Kv_out
  
    R=[]
    np.random.seed(42)
    C=[]; D=[]
    for i in range(n):
        n_clusters = np.random.randint(2, 9)  # 9 exclus → valeurs possibles : 2,3,4,5,6,7,8
        C.append(n_clusters)
        n_space = np.random.randint(2,32)
        D.append(n_space)

    for i in range(n): 
        if dim==0:  p=D[i] 
        else: 
           p=dim
        X,y = SyntheticSet(option=option, n_samples=n_samples, dim = p, 
                           max_dev = max_dev,hamming_distance=hamming_test,
                           p_noise=p_noise, n_clusters=C[i], index_random=i)
        
        if savesyntheticoption == True: #option save synthetic.
            F_name = savesynthethicdata + "/synthetic_" + "d" + str(p) + "-" + str(i) + ".txt" 
            save_xy_to_txt(X, y, filename=F_name)
            
        if Competiteurs == True: R.append(Testcompetitors(X, y, run, noise, noise_methode,
                                                          namemodel=namemodel+".pth",draw = draw))
        
    DisplayMethod(run)    
    if Competiteurs == True:    
        R_2D = np.array(R)
        print("Mutual, ARI, SIL,%,minmax,cl,% recognition")
        all_means = []
        for i in range(0,R_2D.shape[1]):
            means, stds, S = compute_custom_stats(R_2D, i, true_k_index=3, pred_k_index=4)
            print("stats:", " ".join([f"{m:.3f}" for m in means]))
            all_means.append(means)

        # Conversion en array pour écriture propre
        all_means = np.array(all_means)

        # Sauvegarde dans un fichier texte avec tabulations    
        np.savetxt(Saveresult + str(dim)+"d"+str(p_noise)+".txt", all_means, fmt="%.3f", delimiter="\t")    
    
#######################################################################################

def Neighbordhoodmethod(genpoint_s = False,createmodel = False, 
                        hamming_train=0.01,use_fuzzy=True,
                        dev_training =0.3,epoch=200, 
                        dim = range(32, 33), n_setperdimension=10,
                        ourmethod_parameter =[0.4,0.6,0.8],
                        testmodel_s = False,  hamming_test=0.1,max_dev= 0.3,option = 2,dimtest = 2, n_test = 10,
                        p_noise=0.05,testmodel_r = False, T_method_low=0, T_method_up= 1,folder_real_data="realdata",
                        n_samples = 1000,
                        p_embedding = 10, training_data=None, 
                        namemodel = "models/model"+"hybrid2-31"+".pth",
                        Optionsaveresultreal = False,
                        saverealresult = "resultsreal/highdim",
                        savesyntheticresult = "resultsynthetique/all/result",
                        savesynthethicdata = "savesyntheticdata/level1",
                        savesyntheticoption = False,
                        option_model = 0, draw = True):
    
    genpoint_s = genpoint_s
    createmodel = createmodel 
    testmodel_s = testmodel_s  
    testmodel_r = testmodel_r
    n_samples = n_samples
    p_embedding = p_embedding


    if genpoint_s == True:
        for i in range(0,1):
            X,in_pat, out_pat = genpoint(n_samples=n_samples, dim=dim[0], max_dev=0.3,p_noise=p_noise,
                                         p_embedding = 10,Kv_in = 16, 
                                         Kv_out = 16, K_m = 5,n_centers = 4,
                                         Draw = draw, filtered = True, save = True,use_fuzzy=use_fuzzy)

    
    if createmodel == True: 
        CreateModel(n=n_setperdimension,dim=dim, dev_training=dev_training,
                    hamming_training=hamming_train,noise_training = p_noise,
                    p_embedding=p_embedding,training_data = training_data,namemodel=namemodel,
                    option = option_model,epoch=epoch,use_fuzzy=use_fuzzy) 


    if testmodel_s == True:
        TestModelSynthetique(n=n_test ,dim=dimtest,max_dev = max_dev,
                             hamming_test=hamming_test,
                             ourmethod_parameter=ourmethod_parameter,
                             p_embedding=p_embedding,
                             option=option, p_noise=p_noise, 
                             n_samples = n_samples,T_method_low=T_method_low, 
                             T_method_up= T_method_up,namemodel = namemodel,
                             Saveresult = savesyntheticresult,
                             savesynthethicdata = savesynthethicdata,
                             savesyntheticoption = savesyntheticoption,
                             draw = draw)
        
  
    
    if testmodel_r == True:
        nameresult=saverealresult + "/"
        TestModelReal(directory_path = folder_real_data,p_embedding=p_embedding,
                  namemodel=namemodel,n_samples = n_samples,
                  T_method_low=T_method_low, T_method_up= T_method_up,
                  ourmethod_parameter=ourmethod_parameter,
                  Optionsaveresultreal = Optionsaveresultreal,
                  nameresult=nameresult,draw = draw)
        
 
#######################################################################################
def ManuscriptSynthetique():
    v_dev=[0.33]
    n_test_per_dim = 50
    for v_dim in range(2,33):
        p_noise = 0.15
        for noise in range(0,1):
            Neighbordhoodmethod(ourmethod_parameter =[0.6,0.7,0.8,0.9],
                                testmodel_s = True, max_dev= v_dev[0], 
                                option = 0, dimtest = v_dim,n_test = n_test_per_dim,
                                hamming_test=0.05,p_noise=p_noise, 
                                T_method_low=11, T_method_up= 12,
                                namemodel= "models/modelhybrid2-32novel", option_model = 0, 
                                savesyntheticresult = "resultsynthetique/all/result",
                                savesynthethicdata = "savesyntheticdata/level1",
                                savesyntheticoption = False,
                                draw = False)
            p_noise = p_noise + 0.1   

#######################################################################################
def ManuscriptReal():
    
    for methode in range(0,1):
        Neighbordhoodmethod(ourmethod_parameter =[0.5,0.7,0.8,0.9],
                            testmodel_r = True,
                            T_method_low=methode, T_method_up= methode+1,
                            #folder_real_data="Highrealdata/d100/minst",n_samples = 1000,
                            folder_real_data="glass",n_samples = 1000,
                            namemodel= "models/modelhybrid2-32novel",  
                            saverealresult = "resultsreal/mediumdim",
                            Optionsaveresultreal = False,
                            draw = True) 
               

#######################################################################################
def main(cfg: Namespace) -> None:
    
 
    flags = [
        cfg.genpoint_s,
        cfg.createmodel,
        cfg.testmodel_synthetic,
        cfg.testmodel_real,
        cfg.deeplearning_methods,
        cfg.experiments_synthetic,
        cfg.experiments_real
            ]

    if sum(flags) != 1:
        raise ValueError("Error : more than one mode is activated!")
    
    if (cfg.deeplearning_methods == False):
        if cfg.experiments_synthetic == True: ManuscriptSynthetique()
        
        if cfg.experiments_real == True: ManuscriptReal()
        
        if  cfg.experiments_synthetic == False and cfg.experiments_real == False: 
            Neighbordhoodmethod(genpoint_s = cfg.genpoint_s,createmodel = cfg.createmodel,
                        use_fuzzy = cfg.use_fuzzy,
                        ourmethod_parameter = cfg.ourmethod_parameter,
                        dim = range(cfg.dimtraining_start, cfg.dimtraining_end),
                        epoch = cfg.epoch, dev_training=cfg.dev_training,
                        hamming_train=cfg.Hamming_ratio_training,
                        n_setperdimension=cfg.n_set_per_dimension,
                        testmodel_s = cfg.testmodel_synthetic, max_dev= cfg.maxdev_test, 
                        option = cfg.option_test_synthetique, 
                        dimtest = cfg.dimension_test,n_test = cfg.n_synthetic_tests,
                        hamming_test=cfg.Hamming_ratio_test,
                        p_noise=cfg.percentage_noise, testmodel_r = cfg.testmodel_real,
                        T_method_low=cfg.low_method_number, T_method_up= cfg.Up_method_number,
                        folder_real_data = cfg.realdata,n_samples = 1000,
                        p_embedding = cfg.dimension_of_p_embedding, 
                        training_data = cfg.name_training_data,
                        saverealresult= cfg.saverealresult,
                        Optionsaveresultreal = cfg.Optionsaveresultreal,
                        savesyntheticresult=cfg.savesyntheticresult,
                        savesyntheticoption=cfg.savesyntheticoption,
                        savesynthethicdata=cfg.savesynthethicdata,
                        namemodel= cfg.name_model, option_model = cfg.Option_model, 
                        draw = cfg.draw_graphics) 
    else:
        print("Loading the deep learning yaml file to launch deeplearning methods")
        launchdeeplearning()
#######################################################################################

if __name__ == "__main__": 
    useyaml=False

    cfg = get_config()
    if useyaml==True:
        cfg = get_config_yaml("yaml_supervised.txt")
        print(cfg)
    main(cfg)

#######################################################################################

