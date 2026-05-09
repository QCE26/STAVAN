#!/usr/bin/env python3
import numpy as np
import STAVAN_Framework
from STAVAN_Framework import *

#Initialize your quantum circuit
qc = STAVAN(2,approach = 1, t_qubits = 0, chi = 4)

# Optional print statements for debugging
# print("Initial Tableau")
# qc.print_tableu_global(0)
# print()
# qc.debug_inner_products()
# print()


#Apply your quantum circuit gate by gate
#---------------------------------------
qc.apply_single_qubit_gate(H,0)
qc.apply_T_gate(0)
qc.apply_single_qubit_gate(H,0)
qc.apply_single_qubit_gate(H,1)
qc.apply_T_gate(1)
#---------------------------------------


#For Approach 1, directly view and compare the final tableaus for different circuits to check equivalence
#---------------------------------------
qc.print_tableu_global(0)
qc.print_tableu_global(1)
#---------------------------------------


"""
#For Approach 2 and 3, compute the output probabilties for given bitstream. Uncomment the required code lines to test.

#---------------------------------------
p00 = qc.compute_probability_Approach_2([0,1],[0,0])
p01 = qc.compute_probability_Approach_2([0,1],[0,1])
p10 = qc.compute_probability_Approach_2([0,1],[1,0])
p11 = qc.compute_probability_Approach_2([0,1],[1,1])

print(p00,p01,p10,p11)
print(p00+p01+p10+p11)

print()
#---------------------------------------


#---------------------------------------
p00 = qc.compute_probability_Approach_3([0,1], [0,0], num_samples=500)
p01 = qc.compute_probability_Approach_3([0,1], [0,1], num_samples=500)
p10 = qc.compute_probability_Approach_3([0,1], [1,0], num_samples=500)
p11 = qc.compute_probability_Approach_3([0,1], [1,1], num_samples=500)

print(p00,p01,p10,p11)
print(p00+p01+p10+p11)

print()
#---------------------------------------


#---------------------------------------
p0 = qc.compute_marginal_Approach_2(0)
p1 = qc.compute_marginal_Approach_2(1)

print(p0,p1)

print()
#---------------------------------------


#---------------------------------------
p0 = qc.compute_marginal_Approach_3(0, 500)
p1 = qc.compute_marginal_Approach_3(1, 500)

print(p0,p1)
#---------------------------------------


# print()
# print("Final Tableau")
# qc.print_tableu_global(0)
"""