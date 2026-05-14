#!/usr/bin/env python3
import numpy as np
import STAVAN_Framework
from STAVAN_Framework import *
import matplotlib.pyplot as plt
import time
import os

n = 20
t = 30
chi = 500

ckt = 'Vary_param' #Vary_param or Initial_Stabilizer_Decomposition
approach = 3 #Approach 2 or 3
num_samples = 3000

def demo_circuit(qc, n, s):
    # Step 1: Hadamard
    for i in range(n):
        qc.apply_single_qubit_gate(H, i)

    # Step 2: Oracle (simulate using T-heavy structure)
    for i in range(n):
        if s[i] == 1:
            qc.apply_T_gate(i)

    # Step 3: Hadamard again
    for i in range(n):
        qc.apply_single_qubit_gate(H, i)


if(ckt == 'Vary_param'):
    start = time.time()
    print("Start")
    qc = STAVAN(n, t, chi)
    secret = np.random.randint(0, 2, size=n)
    print("Secret:", secret)
    demo_circuit(qc, n, secret)
    
    # measure marginal probabilities of each qubit = |1>
    p = np.zeros(n, dtype=float)
    mse = 0.0

    for i in range(n):
        if(approach == 2):
            p[i] = qc.compute_marginal_Approach_2(i)
        else:
            p[i] = qc.compute_marginal_Approach_3(i,num_samples)
            p[i] = np.maximum(0, p[i])

        #print(i,p[i])

        if secret[i] == 1:
            mse += (p[i] - 0.5)**2
        else:
            mse += (p[i])**2
    end = time.time()
    mse = mse / n
    print("Output:", p)
    print("MSE:", mse)
    print("Time:", end-start)

    #Plot
    x = np.arange(n)
    plt.figure()
    plt.bar(x,p,color="purple")
    plt.bar(x, 1-p,bottom=p,color="yellow")
    plt.xticks(x,secret)
    plt.show()
    save_folder = "plots"
    save_file = str(ckt)+"_"+str(approach)+"_"+str(n)+"_"+str(t)+"_"+str(chi)+"_"+str(num_samples)+".png"
    save_loc = os.path.join(save_folder, save_file)
    plt.savefig(save_loc)

if(ckt == 'Initial_Stabilizer_Decomposition'):
    N_array = [20, 30, 40, 50, 60, 80, 100, 500] #modify based on requirement
    
    for n in N_array:
        start = time.time()
        qc = StabilizerState(n,t,chi)
        end = time.time()
        print("n:", n)
        print("Time:", end-start)
        print()
