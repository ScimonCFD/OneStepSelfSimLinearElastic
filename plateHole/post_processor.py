# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.

#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
#  This is a postprocessing tool. It creates plots for visualising the results
#  from the selfSimulation algorithm. 

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import numpy as np
from auxiliary_functions import deserialise, terminal
from distutils.dir_util import mkpath
import input_file
from input_file import *
from joblib import load
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

mkpath(ROUTE_NN_MODEL + "Results/Plots/")
terminal("cp -r " + ROUTE_NN_MODEL + "loadpass* " + ROUTE_NN_MODEL + 
         "Results/")
master_folder_NN = ROUTE_NN_MODEL + "Results/"

LOAD_INC_INTEREST = 15
TOTAL_LOAD_INCREMENTS =15
TOTAL_NUMBER_PASSES = 10
DELTA_PASSES = 1

sigma_Expected_List = []
epsilon_Expected_List = []
D_Expected_List = []
sigma_A_List = []
sigma_B_List = []
epsilon_A_List = []
epsilon_B_List = []
D_A_List = []
D_B_List = []

# Set the default text font size
plt.rc('font', size=16)# Set the axes title font size
plt.rc('axes', titlesize=20)# Set the axes labels font size
plt.rc('axes', labelsize=20)# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)# Set the legend font size
plt.rc('legend', fontsize=18)# Set the font size of the figure title
plt.rc('figure', titlesize=24)

for i in range(1, TOTAL_NUMBER_PASSES+1, DELTA_PASSES):
    sigma_Expected = deserialise(master_folder_NN + "loadpass" + str(i) + 
                                  "_loadInc" + str(LOAD_INC_INTEREST) + "/" +  
                                  str(LOAD_INC_INTEREST) + "/" , 
                                  "sigmaExpected")
    sigma_Expected_List.append(sigma_Expected)
    
    epsilon_Expected = deserialise(master_folder_NN + "loadpass" + str(i) + 
                                    "_loadInc" + str(LOAD_INC_INTEREST) + "/" + 
                                    str(LOAD_INC_INTEREST) + "/" , 
                                    "epsilonExpected")
    epsilon_Expected_List.append(epsilon_Expected)
    
    D_Expected = deserialise(master_folder_NN + "loadpass" + str(i) + 
                             "_loadInc" + str(LOAD_INC_INTEREST) + "/" +  
                             str(LOAD_INC_INTEREST) + "/" , "DExpected")
    D_Expected_List.append(D_Expected)
    
    sigma_A = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" +
                        str(LOAD_INC_INTEREST) + "/" + 
                        str(LOAD_INC_INTEREST) + "/" , "sigma")
    sigma_A_List.append(sigma_A)

    epsilon_A = deserialise(master_folder_NN + "loadpass" + str(i) + 
                            "_loadInc" + str(LOAD_INC_INTEREST) + "/" + 
                            str(LOAD_INC_INTEREST) + "/" , "epsilon")
    epsilon_A_List.append(epsilon_A)
    
    D_A = deserialise(master_folder_NN + "loadpass" + str(i) + "_loadInc" +
                        str(LOAD_INC_INTEREST) + "/" + 
                        str(LOAD_INC_INTEREST) + "/" , "D")
    D_A_List.append(D_A)

   
epsilon_simul_A_OriginalModel = deserialise(master_folder_NN, 
                                   "epsilon_simul_A_OriginalModel_LoadIncNum0")

sigma_simul_A_OriginalModel = deserialise(master_folder_NN, 
                                     "sigma_simul_A_OriginalModel_LoadIncNum0")

component = ["xx", "xy", "xz", "yy", "yz", "zz"]
for j in range(6):
    # fig = plt.figure(figsize=(15, 10))
    # # plt.scatter(epsilon_Expected_List[0][:,j], 
    # #             epsilon_simul_A_OriginalModel[:,j], color = "red", 
    # #             label = "Calculated", marker='x')
    # plt.plot(epsilon_Expected_List[0][:,j], epsilon_Expected_List[0][:,j], 
    #          color = "blue", label = "Ideal")
    # plt.xlabel(r'Expected strain $(m/m)$')
    # plt.ylabel(r'Calculated strain $(m/m)$')
    # # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
    # #              + ". Base model. Epsilon_" + component[j], fontsize=20)
    # # plt.legend()
    # fig.savefig(master_folder_NN + "Plots/" + "Epsilon_" + component[j] 
    #             + "_BaseModel.png", bbox_inches='tight')
    # plt.close(fig)
    for i in range(TOTAL_NUMBER_PASSES):
        fig = plt.figure(figsize=(15, 10))
        plt.scatter(epsilon_Expected_List[i][:,j], epsilon_A_List[i][:,j], 
                    color = "green", label = "Calculated", marker='o')
        plt.scatter(epsilon_Expected_List[0][:,j], 
                    epsilon_simul_A_OriginalModel[:,j], color = "red", label = 
                    "Original", marker='+')
        plt.plot(epsilon_Expected_List[i][:,j], epsilon_Expected_List[0][:,j], 
                 color = "blue", label = "Ideal", alpha=0.3)
        plt.xlabel(r'Expected strain $(m/m)$')
        plt.ylabel(r'Calculated strain $(m/m)$')
        # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
        #              + ". Pass number " + str(i+1) + ". Epsilon_" 
        #              + component[j], fontsize=20)
        # plt.legend()
        fig.savefig(master_folder_NN + "Plots/" + "Epsilon_" + component[j] + 
                    "_Iter" + str(i+1) + ".png", bbox_inches='tight')
        plt.close(fig)

for j in range(6):
    # fig = plt.figure(figsize=(15, 10))
    # plt.scatter(sigma_Expected_List[0][:,j], sigma_simul_A_OriginalModel[:,j], 
    #             color = "red", label = "Calculated", marker='x')
    # plt.plot(sigma_Expected_List[0][:,j], sigma_Expected_List[0][:,j], 
    #           color = "blue", label = "Ideal")
    # plt.xlabel(r'Expected stress $(Pa)$')
    # plt.ylabel(r'Calculated stress $(Pa)$')
    # # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
    # #              + ". Base model. Sigma_" + component[j], fontsize=20)
    # # plt.legend()
    # fig.savefig(master_folder_NN + "Plots/" + "Sigma_" + component[j] 
    #             + "_BaseModel.png", bbox_inches='tight')    
    plt.close(fig)
    for i in range(TOTAL_NUMBER_PASSES):
        fig = plt.figure(figsize=(15, 10))
        plt.scatter(sigma_Expected_List[i][:,j], sigma_A_List[i][:,j], 
                    color = "green", label = "Calculated", marker='o')
        plt.scatter(sigma_Expected_List[0][:,j], sigma_simul_A_OriginalModel[:,j], 
                color = "red", label = "Original", marker='+')
        plt.plot(sigma_Expected_List[i][:,j], sigma_Expected_List[0][:,j], 
                  color = "blue", label = "Ideal")
        plt.xlabel(r'Expected stress $(Pa)$')
        plt.ylabel(r'Calculated stress $(Pa)$')
        # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
        #              + ". Pass number " + str(i+1) + ". Sigma_" 
        #              + component[j], fontsize=20)
        # plt.legend()
        fig.savefig(master_folder_NN + "Plots/" + "Sigma_" + component[j] 
                    + "_Iter" + str(i+1) + ".png", bbox_inches='tight')
        plt.close(fig)

D_simul_A_OriginalModel_LoadIncNum0 = deserialise(master_folder_NN, 
                                          "D_simul_A_OriginalModel_LoadIncNum0")

component = ["x", "y", "z"]
for j in range(3):
    # fig = plt.figure(figsize=(15, 10))
    # plt.scatter(D_Expected_List[0][:,j]/1e-6, 
    #             D_simul_A_OriginalModel_LoadIncNum0[:,j]/1e-6, color = "red", 
    #             label = "Calculated", marker='x')
    # plt.plot(D_Expected_List[0][:,j]/1e-6, D_Expected_List[0][:,j]/1e-6, 
    #           color = "blue", label = "Ideal")
    # plt.xlabel(r'Expected displacement $(\mu m)$')
    # plt.ylabel(r'Calculated displacement $(\mu m)$')
    # # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
    # #              + ". Base model. D_" + component[j], fontsize=20)
    # # plt.legend()
    # fig.savefig(master_folder_NN + "Plots/" + "D_" + component[j] 
    #             + "_BaseModel.png", bbox_inches='tight')    
    # plt.close(fig)
    for i in range(TOTAL_NUMBER_PASSES):
        fig = plt.figure(figsize=(15, 10))
        plt.scatter(D_Expected_List[i][:,j]/1e-6, D_A_List[i][:,j]/1e-6, 
                    color = "green", label = "Calculated", marker='o')
        plt.scatter(D_Expected_List[0][:,j]/1e-6, 
                D_simul_A_OriginalModel_LoadIncNum0[:,j]/1e-6, color = "red", 
                label = "Original", marker='+')
        plt.plot(D_Expected_List[i][:,j]/1e-6, D_Expected_List[0][:,j]/1e-6, 
                  color = "blue", label = "Ideal")

        plt.xlabel(r'Expected displacement $(\mu m)$')
        plt.ylabel(r'Calculated displacement $(\mu m)$')
        # fig.suptitle("Load increment "+ str(LOAD_INC_INTEREST) 
        #              + ". Pass number " + str(i+1) + ". D_" + component[j], 
        #              fontsize=20)
        # plt.legend()
        fig.savefig(master_folder_NN + "Plots/" + "D_" + component[j] 
                    + "_Iter" + str(i+1) + ".png", bbox_inches='tight')
        plt.close(fig)