from multiprocessing import Process
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from numpy import genfromtxt
from scipy.cluster.hierarchy import fclusterdata
##simulated data with peak at 2.1... MeV
my_data = genfromtxt('data_stream_2121.txt', delimiter=',')
#real data from 60Co source
#my_data = genfromtxt('real_data_co60.txt', delimiter=',')
my_data[:,4] = my_data[:,4]+4500  #this step is needed, I only want positive time values, so that I can use the time as a radius

# ### structure of mydata : eventnr, energy, theta, phi, hit-time

my_data = my_data*[1.,1.,3.14159/180,3.14159/180,1.]

def sph2cart(az, el, r, energy):
    rsin_theta = r*np.sin(el)
    x = rsin_theta*np.cos(az)
    y = rsin_theta*np.sin(az)
    z = r*np.cos(el)
    return x, y, z , energy


def cart2sph(x, y, z, energy):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    th = np.arccos(z/r)
    az = np.arctan2(y,x)
    return th, az, r , energy


#some definitions I need...
def get_center_of_mass_theta_phi_time(np_arr):
    X = 0
    Y = 0
    Z = 0 
    E_sum = 0
    t_mean = 0
    for i in range(np_arr.shape[0]):
        E = np_arr[i,1]
        E_sum += E
        X +=  E*math.sin(np_arr[i,2])*math.cos(np_arr[i,3])
        Y +=  E*math.sin(np_arr[i,2])*math.sin(np_arr[i,3])
        Z +=  E*math.cos(np_arr[i,2])
        t_mean += (np_arr[i,4] - t_mean)/(i+1)
    theta_mean = np.arccos(Z/math.sqrt(X*X+Y*Y+Z*Z))
    phi_mean = np.arctan2(Y,X)
    
    return E_sum,phi_mean,theta_mean,t_mean

num_rows, num_cols = my_data.shape


#f1 = open("file_well_reco_2clusters.txt","w")
with open('file_well_reco_2clusters.txt', 'w') as f1:
    #1)select unique events
    array_unique_events = np.unique(my_data[:,0])
    #these were the best values for clustering with agglomerative model
    distance_weight = 3640
    time_weight = 2.5
    #2)loop over those events (2 in a row)
    for i in range(0,(len(array_unique_events)-2),2):
    #3)if we have exactly two clusters, correctly clusterized-> get for both clusters there center of mass etc
        E1 = my_data[my_data[:,0] == array_unique_events[i]]
        E2 = my_data[my_data[:,0] == array_unique_events[i+1]]
        E_1 = E1[:,[2,3,4,1]]
        E_2 = E2[:,[2,3,4,1]]
        cart_e1 = sph2cart(E_1[:,1],E_1[:,0],time_weight*E_1[:,2],E_1[:,3])
        np_cart_e1 = np.asarray(cart_e1)
        np_cart_e1 = np.transpose(np_cart_e1)
        
        cart_e2 = sph2cart(E_2[:,1],E_2[:,0],time_weight*E_2[:,2],E_2[:,3])
        np_cart_e2 = np.asarray(cart_e2)
        np_cart_e2 = np.transpose(np_cart_e2)
        arr_energy1 = np_cart_e1[:,3]
        arr_energy1 = arr_energy1.flatten()
        arr_energy2 = np_cart_e2[:,3]
        arr_energy2 = arr_energy2.flatten()
        list_of_energies = [arr_energy1,arr_energy2]
        list_of_energies = [l.tolist() for l in list_of_energies]
        anal_list_of_energies = list_of_energies.copy()
        X = np.vstack([np_cart_e1,np_cart_e2])
        data = pd.DataFrame(X, columns = ['x','y','z','energy'])
        data = data.iloc[:,0:3]
        my_np_data = data.to_numpy()
        output = fclusterdata(data, t=distance_weight, criterion='distance',method="ward")
    
        nr_clusters = len(np.unique(output))    
        array_energy_cluster = np.zeros(nr_clusters)
        arr_reconstructed_clusters = [[] for _ in range(nr_clusters)]
        for i in range(X.shape[0]):
                array_energy_cluster[output[i]-1] += X[i,3]
                arr_reconstructed_clusters[output[i]-1].append(X[i,3])
        
        anal_arr_reconstructed_clusters = arr_reconstructed_clusters.copy()
        for i in range(len(arr_reconstructed_clusters)):
                for j in range(2):
                        if (sorted(arr_reconstructed_clusters[i]) == sorted(list_of_energies[j])):
                                #remove all well reconstructed energies, by setting value to -1
                                anal_arr_reconstructed_clusters[i] = [-1]
                                anal_list_of_energies[j] = [-1]
        
        left_over_list_of_energy = [a  for a in anal_list_of_energies if a[0] != -1]
        left_over_arr_reconstructed_clusters = [a  for a in anal_arr_reconstructed_clusters if a != -1]
        not_reconstructed = len(left_over_list_of_energy)
    
        if (not_reconstructed == 0):
            #those are now good events
            Esum_1,phi_m1,theta_m1,time_m1 = get_center_of_mass_theta_phi_time(E1)
            Esum_2,phi_m2,theta_m2,time_m2 = get_center_of_mass_theta_phi_time(E2) 
            delta_phi = abs(phi_m1-phi_m2) 
            delta_theta = abs(theta_m1-theta_m2) 
            delta_time = abs(time_m1-time_m2) 
            #now simply write everything into one line in a file
            #f1.write(Esum_1,phi_m1,theta_m1,time_m1,Esum_2,phi_m2,theta_m2,time_m2,time_m2,delta_theta,delta_phi,delta_time,0 )
            f1.write(repr(Esum_1)+","+repr(phi_m1)+","+repr(theta_m1)+","+repr(time_m1)+","+repr(Esum_2)+","+repr(phi_m2)+","+repr(theta_m2)+","+repr(time_m2)+","+repr(delta_theta)+","+repr(delta_phi)+","+repr(delta_time)+","+"0"+"\n")
    
f1.close()




##now to the second file, where we wrongly create 2 clusters instead of one


with open('file_wrongly_reco_two_clusters.txt', 'w') as f2:
    #1)select unique events
    array_unique_events = np.unique(my_data[:,0])
    #these were the best values for clustering with agglomerative model
    distance_weight = 3640
    time_weight = 2.5
    #2)loop over those events (2 in a row)
    for i in range(0,(len(array_unique_events)-1),1):
    #3)if we have exactly two clusters, correctly clusterized-> get for both clusters there center of mass etc
        E1 = my_data[my_data[:,0] == array_unique_events[i]]
        E_1 = E1[:,[2,3,4,1]]
        cart_e1 = sph2cart(E_1[:,1],E_1[:,0],time_weight*E_1[:,2],E_1[:,3])
        np_cart_e1 = np.asarray(cart_e1)
        np_cart_e1 = np.transpose(np_cart_e1)
        
        arr_energy1 = np_cart_e1[:,3]
        arr_energy1 = arr_energy1.flatten()
        list_of_energies = [arr_energy1]
        list_of_energies = [l.tolist() for l in list_of_energies]
        anal_list_of_energies = list_of_energies.copy()
        X = np.vstack([np_cart_e1])
        data = pd.DataFrame(X, columns = ['x','y','z','energy'])
        #print(data)
        data = data.iloc[:,0:3]
        #my_np_data = data.to_numpy()
        if (data.shape[0] > 1):
            #print(data)
            output = fclusterdata(data, t=distance_weight, criterion='distance',method="ward")
            if (len(np.unique(output)) == 2):
                #create two numpy arrays for the two clusters
                list_cluster1 = []
                list_cluster2 = []
                for i in range(len(output)):
                    
                    if (output[i] == 1):
                        list_cluster1.append((E1[i,:]).tolist())
                        print("First cluster, hit:")
                        print((E1[i,:]))
                    if (output[i] == 2):
                        print("Second cluster, hit:")
                        list_cluster2.append((E1[i,:]).tolist())
                        print((E1[i,:]))
                np_arr_cluster1 = np.array(list_cluster1)
                np_arr_cluster2 = np.array(list_cluster2)
                print(np_arr_cluster1)
                print(np_arr_cluster2)
                Esum_1,phi_m1,theta_m1,time_m1 = get_center_of_mass_theta_phi_time(np_arr_cluster1)
                Esum_2,phi_m2,theta_m2,time_m2 = get_center_of_mass_theta_phi_time(np_arr_cluster2) 
                delta_phi = abs(phi_m1-phi_m2) 
                delta_theta = abs(theta_m1-theta_m2) 
                delta_time = abs(time_m1-time_m2) 
                f2.write(repr(Esum_1)+","+repr(phi_m1)+","+repr(theta_m1)+","+repr(time_m1)+","+repr(Esum_2)+","+repr(phi_m2)+","+repr(theta_m2)+","+repr(time_m2)+","+repr(delta_theta)+","+repr(delta_phi)+","+repr(delta_time)+","+"1"+"\n")




f2.close()




