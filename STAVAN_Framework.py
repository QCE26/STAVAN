import numpy as np
from itertools import product
from typing import List, Tuple
import random

X = np.array(([0,1],[1,0]))
Z = np.array(([1,0],[0,-1]))
Y = np.array(([0,-1j],[1j,0]))
I = np.array(([1,0],[0,1]))
H = np.array(([1,1],[1,-1]))/np.sqrt(2)
S = np.array(([1,0],[0,1j]))
S_dg = np.array(([1,0],[0,-1j]))
T = np.array(([1,0],[0,np.exp(1j * np.pi / 4)]))
CNOT = np.array(([1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]))

# Pauli list and code
paulis = {
    "I": (I, [0, 0]),
    "Z": (Z, [0, 1]),
    "X": (X, [1, 0]),
    "Y": (Y, [1, 1])
}

#Stabilizer Tableau
class STAVAN:
    def __init__(self, n_qubits, approach = 3, t_qubits = 0, chi = 2):
        self.n_qubits_og = n_qubits #Number of qubits in the Quantum Circuit
        self.t_qubits = t_qubits #Number of T gates
        self.n_qubits = n_qubits+t_qubits
        self.t_count = 0
        self.chi = chi
        self.approach = approach
        self.inner_cache = {}
        self.pauli_cache = {}

        if t_qubits == 0 or approach == 1:
            self.approach = 1
            self.t_qubits = 0
            self.n_qubits = n_qubits

        tableu0 = np.zeros((n_qubits, 2*n_qubits), dtype = int)
        phases0 = np.zeros((n_qubits), dtype = int)
        tableu1 = np.zeros((n_qubits, 2*n_qubits), dtype = int)
        phases1 = np.zeros((n_qubits), dtype = int)

        #As per Bravyi Gosset, phases will be iota^n (n E Natural Numbers)
        #We represent phases by 0,1,2,3 (2 bits)

        #put in |000...0> state
        #Stabilizer State Tableu: III...IZI....III where for the kth qubit we put Z at kth position and rest all I (k = 1..n_qubits)
        for i in range(n_qubits):
            tableu0[i][2*i + 1] = 1

        #put in |+++...+> state
        #Stabilizer State Tableu: III...IXI....III where for the kth qubit we put X at kth position and rest all I (k = 1..n_qubits)
        for i in range(n_qubits):
            tableu1[i][2*i] = 1

        
        
        if t_qubits > 0 and (approach == 2 or approach == 3):
            #Now decompose |A> into χ stabilizer states, generated using random circuit generator. The amplitude will be equal to the overlap between |A> and the random stabilizer state.
        
            self.global_tableu0 = []
            self.global_tableu1 = []

            magic_state_decomp = self.Ancilla_decomposition(t_qubits)
        
            if t_qubits > 0:
                for i in range(chi):
                    #tableu_t, phases_t, ampl_t = self.generate_random_stabilizer_state(t_qubits)
                    tableu_t, phases_t, ampl_t = magic_state_decomp[i]
                    
                    new_tableu0 = np.block([[tableu0, np.zeros((tableu0.shape[0], tableu_t.shape[1]), dtype = int)],[np.zeros((tableu_t.shape[0], tableu0.shape[1]), dtype = int),tableu_t]])
                    new_phases0 = np.concatenate((phases0, phases_t))

                    new_tableu1 = np.block([[tableu1, np.zeros((tableu1.shape[0], tableu_t.shape[1]), dtype = int)],[np.zeros((tableu_t.shape[0], tableu1.shape[1]), dtype = int),tableu_t]])
                    new_phases1 = np.concatenate((phases1, phases_t))

                    self.global_tableu0.append([new_tableu0, new_phases0, ampl_t])
                    self.global_tableu1.append([new_tableu1, new_phases1, ampl_t])
                
                # normalize coefficients
                norm = np.sqrt(sum(abs(ampl)**2 for _, _, ampl in self.global_tableu0))

                for i in range(len(self.global_tableu0)):
                    self.global_tableu0[i][2] /= norm

                for i in range(len(self.global_tableu1)):
                    self.global_tableu1[i][2] /= norm


        else:
            self.global_tableu0 = [[tableu0, phases0, 1]]
            self.global_tableu1 = [[tableu1, phases1, 1]]

        #print("Initial tableau representing |000...0> state:")
        #self.print_tableu_global(0)

        #print("Initial tableau representing |+++...+> state:")
        #self.print_tableu_global(1)
    
    def split_tableau(self, tableau):
        X = tableau[:, 0::2].copy()
        Z = tableau[:, 1::2].copy()
        return X, Z
    
    def to_array(self, quantum_state):
        if(quantum_state == [0,1]):
                return Z
        elif(quantum_state == [1,0]):
                return X
        elif(quantum_state == [1,1]):
                return Y
        elif(quantum_state == [0,0]):
                return I
    
    def commute(self, A, B, tol=1e-10):
        """Return True if A and B commute, i.e. AB == BA (within tolerance)."""
        AB = np.matmul(A,B)
        BA = np.matmul(B,A)
        return np.allclose(AB, BA, atol=tol)
    
    def update_pauli_entry(self, quantum_state):
        
        if np.allclose(Z, quantum_state):
            return 0,1,1
        elif np.allclose(-Z, quantum_state):
            return 0,1,-1
        if np.allclose(Z*1j, quantum_state):
            return 0,1,1j
        if np.allclose(-Z*1j, quantum_state):
            return 0,1,-1j
        if np.allclose(X, quantum_state):
            return 1,0,1
        elif np.allclose(-X, quantum_state):
            return 1,0,-1
        if np.allclose(X*1j, quantum_state):
            return 1,0,1j
        if np.allclose(-X*1j, quantum_state):
            return 1,0,-1j
        if np.allclose(Y, quantum_state):
            return 1,1,1
        elif np.allclose(-Y, quantum_state):
            return 1,1,-1
        if np.allclose(Y*1j, quantum_state):
            return 1,1,1j
        if np.allclose(-Y*1j, quantum_state):
            return 1,1,-1j
        if np.allclose(I, quantum_state):
            return 0,0,1
        elif np.allclose(-I, quantum_state):
            return 0,0,-1
        if np.allclose(I*1j, quantum_state):
            return 0,0,1j
        if np.allclose(-I*1j, quantum_state):
            return 0,0,-1j
        return None, None, None

    def print_tableu_global(self, tableu_number):
        if(tableu_number == 0):
            print("|000...0> state:")
            for tableu, phases, amplitude in self.global_tableu0:
                print(f"Amplitude: {amplitude:.4f}")
                for i in range(self.n_qubits):
                    if(phases[i] == 0):
                        st = "+  "
                    elif(phases[i] == 1):
                        st = "+j "
                    elif(phases[i] == 2):
                        st = "-  "
                    else:
                        st = "-j "
                    
                    for j in range(self.n_qubits):
                        k = [tableu[i][2*j], tableu[i][2*j+1]]
                        if(k == [0,0]):
                            st += "I "
                        elif(k == [0,1]):
                            st += "Z "
                        elif(k == [1,0]):
                            st += "X "
                        elif(k == [1,1]):
                            st += "Y "
                    
                    print(st)
        else:
            print("|+++...+> state:")
            for tableu, phases, amplitude in self.global_tableu1:
                print(f"Amplitude: {amplitude:.4f}")
                for i in range(self.n_qubits):
                    if(phases[i] == 0):
                        st = "+  "
                    elif(phases[i] == 1):
                        st = "+j "
                    elif(phases[i] == 2):
                        st = "-  "
                    else:
                        st = "-j "
                    
                    for j in range(self.n_qubits):
                        k = [tableu[i][2*j], tableu[i][2*j+1]]
                        if(k == [0,0]):
                            st += "I "
                        elif(k == [0,1]):
                            st += "Z "
                        elif(k == [1,0]):
                            st += "X "
                        elif(k == [1,1]):
                            st += "Y "
                    
                    print(st)
        return
    
    def hash_state(self, X, Z, p):
        return (
            X.tobytes(),
            Z.tobytes(),
            p.tobytes()
        )
    
    def generate_random_stabilizer_state(self, t):
        tableau = np.zeros((t, 2*t), dtype=int)
        phases = np.zeros(t, dtype=int)

        # start from |0^t>
        for i in range(t):
            tableau[i][2*i+1] = 1  # Z stabilizer

        # apply random Clifford circuit
        num_gates = 10 * t

        for _ in range(num_gates):
            gate_type = random.choice(["H", "S", "CNOT"])

            for n in range(t):
                if gate_type == "H":
                    qubit_index = random.randint(0, t-1)
                    x = tableau[n][2*qubit_index]
                    z = tableau[n][2*qubit_index+1]
                    p = phases[n]

                    if x == 1 and z == 1:
                        phases[n] = (p+2)%4
                    
                    tableau[n][2*qubit_index] = z
                    tableau[n][2*qubit_index+1] = x

                elif gate_type == "S":
                    qubit_index = random.randint(0, t-1)
                    x = tableau[n][2*qubit_index]
                    z = tableau[n][2*qubit_index+1]
                    p = phases[n]

                    if x == 1 and z == 0:
                        tableau[n][2*qubit_index+1] = 1
                    elif x == 1 and z == 1:
                        tableau[n][2*qubit_index+1] = 0
                        phases[n] = (p+2)%4

                elif gate_type == "CNOT":
                    c_bit = random.randint(0, t-1)
                    t_bit = random.randint(0, t-1)
                    if c_bit != t_bit:
                        x_c = tableau[n][2*c_bit]
                        z_c = tableau[n][2*c_bit+1]

                        x_t = tableau[n][2*t_bit]
                        z_t = tableau[n][2*t_bit+1]

                        if x_c == 1 and z_t == 1 and (x_t ^ z_c ^ 1):
                            phases[n] = (phases[n] + 2) % 4
                    
                        tableau[n][2*t_bit] ^= x_c
                        tableau[n][2*c_bit+1] ^= z_t
                        

        return tableau, phases
    
    def compute_magic_overlap(self, tableau, phases, t):

        X = tableau[:, 0::2]
        Z = tableau[:, 1::2]

        overlap = 1.0 + 0j

        for i in range(t):

            # check stabilizer on qubit i
            has_X = np.any(X[:, i])
            has_Z = np.any(Z[:, i])

            if has_X and not has_Z:
                # behaves like |+>
                overlap *= np.cos(np.pi/8)

            elif has_X and has_Z:
                # behaves like |Y>
                overlap *= np.exp(1j*np.pi/4) * np.sin(np.pi/8)

            elif has_Z and not has_X:
                # behaves like |0>
                overlap *= 1 / np.sqrt(2)

            else:
                overlap *= 1 / np.sqrt(2)

        return overlap
    
    def Ancilla_decomposition(self, t):
        #Decomposition of |A>^t into chi stabilizer states. We use oversampling + importance selection to avoid zero-overlap issue.

        OVERSAMPLE = 8   #increase if unstable (5–10 works well)

        candidates = []

        # Oversample stabilizers
        for _ in range(self.chi * OVERSAMPLE):
            tab, phases = self.generate_random_stabilizer_state(t)
            coeff = self.compute_magic_overlap(tab, phases, t)
            weight = abs(coeff)**2
            candidates.append((tab, phases, coeff, weight))

        # Handle pathological cases
        weights = np.array([w for _,_,_,w in candidates])

        if weights.sum() < 1e-14:
            states = []
            for _ in range(self.chi):
                tab, phases = self.generate_random_stabilizer_state(t)
                states.append([tab, phases, 1.0 + 0j])
            norm = np.sqrt(self.chi)
            for s in states:
                s[2] /= norm
            return states

        # Select top-χ stabilizers by importance (largest overlap)
        candidates.sort(key=lambda x: -x[3])
        selected = candidates[:self.chi]

        # Extract states
        states = []
        for tab, phases, coeff, _ in selected:
            states.append([tab, phases, coeff])
        norm = np.sqrt(sum(abs(c)**2 for _,_,c in states))
        if norm < 1e-14:
            return self.Ancilla_decomposition(t)
        for s in states:
            s[2] /= norm

        return states
    
    def apply_single_qubit_gate(self, gate, qubit_index):
        #print()
        #print("############################################################")
        #print("Apply " + str(gate) + " to " + str(qubit_index))
        # print("Original tableau:")
        # self.print_tableu_global(0)
        # print()
        # self.print_tableu_global(1)
        #print()


        #For |000...0> state
        for i in range(len(self.global_tableu0)):
            tableu0, phases0, _ = self.global_tableu0[i]

            #Modify each row of tableu
            for n in range(self.n_qubits):
                x = tableu0[n][2*qubit_index]
                z = tableu0[n][2*qubit_index+1]
                p = phases0[n]

                if np.allclose(gate, H):
                    if x == 1 and z == 1: #Y --> -Y
                        phases0[n] = (p+2)%4
                    #X <--> Z
                    tableu0[n][2*qubit_index] = z
                    tableu0[n][2*qubit_index+1] = x
                
                elif np.allclose(gate, S):
                    if x == 1 and z == 0: #X --> Y
                        tableu0[n][2*qubit_index+1] = 1
                    elif x == 1 and z == 1: #Y --> -X
                        tableu0[n][2*qubit_index+1] = 0
                        phases0[n] = (p+2)%4
                
                elif np.allclose(gate, X):
                    if z == 1:
                        phases0[n] = (p+2)%4
                
                elif np.allclose(gate, Z):
                    if x == 1:
                        phases0[n] = (p+2)%4
                
                elif np.allclose(gate, Y):
                    if x != z:
                        phases0[n] = (p+2)%4
                
                elif np.allclose(gate, S_dg):
                    if x == 1 and z == 0: #X --> -Y
                        tableu0[n][2*qubit_index+1] = 1
                        phases0[n] = (p+2)%4
                    elif x == 1 and z == 1: #Y --> X
                        tableu0[n][2*qubit_index+1] = 0

            self.global_tableu0[i][0] = tableu0
            self.global_tableu0[i][1] = phases0
        
        
        #For |+++...+> state
        for i in range(len(self.global_tableu1)):
            tableu1, phases1, _ = self.global_tableu1[i]

            #Modify each row of tableu
            for n in range(self.n_qubits):
                x = tableu1[n][2*qubit_index]
                z = tableu1[n][2*qubit_index+1]
                p = phases1[n]

                if np.allclose(gate, H):
                    if x == 1 and z == 1: #Y --> -Y
                        phases1[n] = (p+2)%4
                    #X <--> Z
                    tableu1[n][2*qubit_index] = z
                    tableu1[n][2*qubit_index+1] = x
                
                elif np.allclose(gate, S):
                    if x == 1 and z == 0: #X --> Y
                        tableu1[n][2*qubit_index+1] = 1
                    elif x == 1 and z == 1: #Y --> -X
                        tableu1[n][2*qubit_index+1] = 0
                        phases1[n] = (p+2)%4
                
                elif np.allclose(gate, X):
                    if z == 1:
                        phases1[n] = (p+2)%4
                
                elif np.allclose(gate, Z):
                    if x == 1:
                        phases1[n] = (p+2)%4
                
                elif np.allclose(gate, Y):
                    if x != z:
                        phases1[n] = (p+2)%4
                
                elif np.allclose(gate, S_dg):
                    if x == 1 and z == 0: #X --> -Y
                        tableu1[n][2*qubit_index+1] = 1
                        phases1[n] = (p+2)%4
                    elif x == 1 and z == 1: #Y --> X
                        tableu1[n][2*qubit_index+1] = 0
            
            self.global_tableu1[i][0] = tableu1
            self.global_tableu1[i][1] = phases1
        

        #print("-----------------------------------------------------------")
        #print("Updated tableau:")
        #self.print_tableu_global(0)
        #print()
        #self.print_tableu_global(1)
    
    
    def apply_T_gate(self, qubit_index):
        # print()
        # print("############################################################")
        # print("Apply T gate to " + str(qubit_index))
        # print("Original tableau:")
        # self.print_tableu_global(0)
        # print()
        # self.print_tableu_global(1)
        # print()
        
        if self.approach == 1:
            alpha = 0.5*(1 + np.exp(1j*np.pi/4))
            beta = 0.5*(1 - np.exp(1j*np.pi/4))
            
            #For |000...0> state
            global_tableu0_new = []
            for tableu0, phases0, amp0 in self.global_tableu0:
                #We add our ancilla state |A> = (|0> + exp(ipi/4)|1>)/sqrt(2) = a|+> + b|->
                
                #Reconstruct New Tableu along with our Ancilla qubit.
                #For earlier rows, simply add I to the last column.
                #Final row will have: III...IX and phase will have + or - depending on state of Ancilla Qubit breakdown (|+> or |->)
                
                tableu0_1 = np.zeros((self.n_qubits+1, (self.n_qubits+1)*2), dtype = int)
                phases0_1 = np.ones(self.n_qubits+1, dtype = complex)
                tableu0_2 = np.zeros((self.n_qubits+1, (self.n_qubits+1)*2), dtype = int)
                phases0_2 = np.ones(self.n_qubits+1, dtype = complex)

                tableu0_1[:self.n_qubits, :self.n_qubits*2] = tableu0
                tableu0_2[:self.n_qubits, :self.n_qubits*2] = tableu0
                phases0_1[:self.n_qubits] = phases0
                phases0_2[:self.n_qubits] = phases0
                
                tableu0_1[self.n_qubits, (self.n_qubits)*2] = 1

                tableu0_2[self.n_qubits, (self.n_qubits)*2] = 1

                phases0_1[self.n_qubits] = 1
                phases0_2[self.n_qubits] = -1

                amp0_1 = amp0 * alpha
                amp0_2 = amp0 * beta

                global_tableu0_new.append([tableu0_1, phases0_1, amp0_1])
                global_tableu0_new.append([tableu0_2, phases0_2, amp0_2])

            
            #For |+++...+> state
            global_tableu1_new = []
            for tableu0, phases0, amp0 in self.global_tableu1:
                #We add our ancilla state |A> = (|0> + exp(ipi/4)|1>)/sqrt(2) = a|+> + b|->

                #Reconstruct New Tableu along with our Ancilla qubit.
                #For earlier rows, simply add I to the last column.
                #Final row will have: III...IX and phase will have + or - depending on state of Ancilla Qubit breakdown (|+> or |->)
                
                tableu0_1 = np.zeros((self.n_qubits+1, (self.n_qubits+1)*2), dtype = int)
                phases0_1 = np.ones(self.n_qubits+1, dtype = complex)
                tableu0_2 = np.zeros((self.n_qubits+1, (self.n_qubits+1)*2), dtype = int)
                phases0_2 = np.ones(self.n_qubits+1, dtype = complex)

                tableu0_1[:self.n_qubits, :self.n_qubits*2] = tableu0
                tableu0_2[:self.n_qubits, :self.n_qubits*2] = tableu0
                phases0_1[:self.n_qubits] = phases0
                phases0_2[:self.n_qubits] = phases0
                
                tableu0_1[self.n_qubits, (self.n_qubits)*2] = 1

                tableu0_2[self.n_qubits, (self.n_qubits)*2] = 1

                phases0_1[self.n_qubits] = 1
                phases0_2[self.n_qubits] = -1

                amp0_1 = amp0 * alpha
                amp0_2 = amp0 * beta

                global_tableu1_new.append([tableu0_1, phases0_1, amp0_1])
                global_tableu1_new.append([tableu0_2, phases0_2, amp0_2])
            
            self.global_tableu0 = global_tableu0_new
            self.global_tableu1 = global_tableu1_new

            self.n_qubits += 1

            # print("Added Ancillia State.\n Updated tableau:")
            # self.print_tableu_global(0)
            # print()
            # self.print_tableu_global(1)
            # print()


            # print("Applying CNOT gate")
            self.apply_CNOT_gate(qubit_index, self.n_qubits_og + self.t_count)

            #Now, we have to measure the ancilla qubit corresponding to the T gate
            #By construction, the last row is +- III...IX
            #Using measurment operator P = III...IZ
            #Last row (Sn) will anti-commute. We will replace last row by P.
            #Now, we check all other rows. If any of them (Sk) anticommutes with P, replace Sk <-- Sk . Sn
            #We are purposely post selecting 0 as of now.
            
            
            #print("Post Selecting 0 after measurement")
            
            #For |000...0> state
            for i in range(len(self.global_tableu0)):
                tableu0, phases0, amp0 = self.global_tableu0[i]

                tableu0[self.n_qubits-1][(self.n_qubits-1)*2] = 0
                tableu0[self.n_qubits-1][(self.n_qubits)*2 - 1] = 1

                ph_n = phases0[self.n_qubits - 1]
                phases0[self.n_qubits - 1] = 1
                
                #Modify each row of tableu
                for n in range(self.n_qubits - 1):
                    quantum_state = self.to_array([tableu0[n][2*(self.n_qubits-1)], tableu0[n][2*(self.n_qubits)-1]])
                    if self.commute(quantum_state, Z) == False: #Anticommute
                        quantum_state = np.round(np.matmul(quantum_state, X))
                    
                        a,b,ph = self.update_pauli_entry(quantum_state)

                        tableu0[n][2*(self.n_qubits-1)] = a
                        tableu0[n][2*(self.n_qubits)-1] = b
                        phases0[n] *= ph * ph_n

                self.global_tableu0[i][0] = tableu0
                self.global_tableu0[i][1] = phases0
            
            
            #For |+++...+> state
            for i in range(len(self.global_tableu1)):
                tableu1, phases1, _ = self.global_tableu1[i]

                tableu1[self.n_qubits-1][(self.n_qubits-1)*2] = 0
                tableu1[self.n_qubits-1][(self.n_qubits)*2 - 1] = 1

                ph_n = phases1[self.n_qubits - 1]
                phases1[self.n_qubits - 1] = 1
                
                #Modify each row of tableu
                for n in range(self.n_qubits - 1):
                    quantum_state = self.to_array([tableu1[n][2*(self.n_qubits-1)], tableu1[n][2*(self.n_qubits)-1]])
                    if self.commute(quantum_state, Z) == False: #Anticommute
                        quantum_state = np.round(np.matmul(quantum_state, X))
                    
                        a,b,ph = self.update_pauli_entry(quantum_state)
                        tableu1[n][2*(self.n_qubits-1)] = a
                        tableu1[n][2*(self.n_qubits)-1] = b
                        phases1[n] *= ph * ph_n

                self.global_tableu1[i][0] = tableu1
                self.global_tableu1[i][1] = phases1
            

        elif self.approach == 2 or self.approach == 3:
            self.apply_CNOT_gate(qubit_index, self.n_qubits_og + self.t_count)
            self.t_count += 1

        # print("-----------------------------------------------------------")
        # print("Updated tableau:")
        # self.print_tableu_global(0)
        # print()
        # self.print_tableu_global(1)
        # print()
    
    def apply_CNOT_gate(self, c_bit, t_bit):
        #print()
        #print("############################################################")
        #print("Apply CNOT to " + str(t_bit) + " controlled by " + str(c_bit))
        #print("Original tableau:")
        #self.print_tableu_global(0)
        #print()
        #self.print_tableu_global(1)
        #print()


        #For |000...0> state
        for i in range(len(self.global_tableu0)):
            tableu0, phases0, _ = self.global_tableu0[i]

            for n in range(self.n_qubits):
                x_c = tableu0[n][2*c_bit]
                z_c = tableu0[n][2*c_bit+1]

                x_t = tableu0[n][2*t_bit]
                z_t = tableu0[n][2*t_bit+1]

                if x_c == 1 and z_t == 1 and (x_t ^ z_c ^ 1):
                    phases0[n] = (phases0[n] + 2) % 4
                
                #tableu update
                tableu0[n][2*t_bit] ^= x_c
                tableu0[n][2*c_bit+1] ^= z_t
            
            self.global_tableu0[i][0] = tableu0
            self.global_tableu0[i][1] = phases0


        #For |+++...+> state
        for i in range(len(self.global_tableu1)):
            tableu1, phases1, _ = self.global_tableu1[i]

            for n in range(self.n_qubits):
                x_c = tableu1[n][2*c_bit]
                z_c = tableu1[n][2*c_bit+1]

                x_t = tableu1[n][2*t_bit]
                z_t = tableu1[n][2*t_bit+1]

                if x_c == 1 and z_t == 1 and (x_t ^ z_c ^ 1):
                    phases1[n] = (phases1[n] + 2) % 4
                
                #tableu update
                tableu1[n][2*t_bit] ^= x_c
                tableu1[n][2*c_bit+1] ^= z_t

            self.global_tableu1[i][0] = tableu1
            self.global_tableu1[i][1] = phases1
        
        #print("-----------------------------------------------------------")
        #print("Updated tableau:")
        #self.print_tableu_global(0)
        #self.print_tableu_global(1)
        #print()

    def apply_CZ_gate(self, c_bit, t_bit):
        self.apply_single_qubit_gate(H, t_bit)
        self.apply_CNOT_gate(c_bit, t_bit)
        self.apply_single_qubit_gate(H, t_bit)
    
    def apply_pauli_Z(self, tableau, phases, z_mask):
        #Apply Z(s) to stabilizer

        X, Z = self.split_tableau(tableau)
        p = phases.copy()

        n = X.shape[0]

        for i in range(n):
            comm = np.dot(X[i], z_mask) % 2

            if comm == 1:
                p[i] = (p[i] + 2) % 4 #multiply by -1

        return tableau.copy(), p
    
    def pauli_overlap_tableau_cached(self, state_a, state_b, z_mask, measured_qubits):
        key = (
            id(state_a),
            id(state_b),
            tuple(z_mask)
        )

        if key in self.pauli_cache:
            return self.pauli_cache[key]

        val = self.pauli_overlap_tableau_fast(
            state_a, state_b, z_mask, measured_qubits
        )

        self.pauli_cache[key] = val
        return val
    
    def pauli_overlap_tableau_fast(self, state_a, state_b, z_mask, measured_qubits):
        Xa = state_a[0][:, 0::2]
        Za = state_a[0][:, 1::2]
        pa = state_a[1]

        Xb = state_b[0][:, 0::2].copy()
        Zb = state_b[0][:, 1::2].copy()
        pb = state_b[1].copy()

        for i in range(Xb.shape[0]):
            comm = 0
            for q in measured_qubits:
                comm ^= (Xb[i, q] & z_mask[q])

            if comm == 1:
                pb[i] = (pb[i] + 2) % 4

        return self.stabilizer_inner_product_cached(Xa, Za, pa, Xb, Zb, pb)
    
    def stabilizer_inner_product_cached(self, Xa, Za, pa, Xb, Zb, pb):
        key = (
            self.hash_state(Xa, Za, pa),
            self.hash_state(Xb, Zb, pb)
        )

        if key in self.inner_cache:
            return self.inner_cache[key]

        val = self.stabilizer_inner_product_fast(Xa, Za, pa, Xb, Zb, pb)

        self.inner_cache[key] = val
        return val
    
    def stabilizer_inner_product_fast(self, Xa, Za, pa, Xb, Zb, pb):
        n = Xa.shape[0]
        log2_overlap = 0

        for i in range(n):
            x = Xa[i]
            z = Za[i]
            phase_a = pa[i]
            matched = False
            for j in range(n):
                xb = Xb[j]
                zb = Zb[j]
                phase_b = pb[j]
                if np.array_equal(x, xb) and np.array_equal(z, zb):
                    if (phase_a - phase_b) % 4 == 2:
                        return 0.0
                    matched = True
                    break
            if matched:
                continue

            # check if it anticommutes with ANY generator in B
            anticommutes = False
            for j in range(n):
                xb = Xb[j]
                zb = Zb[j]
                comm = (np.dot(x, zb) + np.dot(z, xb)) % 2
                if comm == 1:
                    anticommutes = True
                    break
            if anticommutes:
                return 0.0
            log2_overlap += 1

        return 2 ** (-log2_overlap / 2)
    
    def debug_inner_products(self):
        states = self.global_tableu0

        for i in range(len(states)):
            for j in range(len(states)):
                Xi = states[i][0][:, 0::2]
                Zi = states[i][0][:, 1::2]
                pi = states[i][1]

                Xj = states[j][0][:, 0::2]
                Zj = states[j][0][:, 1::2]
                pj = states[j][1]

                val = self.stabilizer_inner_product_cached(Xi, Zi, pi, Xj, Zj, pj)

                print(f"<{i}|{j}> = {val}")
    
    def precompute_gram_matrix(self):
        states = self.global_tableu0
        chi = len(states)

        G = np.zeros((chi, chi), dtype=complex)

        for a in range(chi):
            Xa = states[a][0][:, 0::2]
            Za = states[a][0][:, 1::2]
            pa = states[a][1]

            for b in range(chi):
                Xb = states[b][0][:, 0::2]
                Zb = states[b][0][:, 1::2]
                pb = states[b][1]

                G[a, b] = self.stabilizer_inner_product_fast(
                    Xa, Za, pa, Xb, Zb, pb
                )

        return G
    
    def compute_probability_Approach_2(self, measured_qubits, x):

        states = self.global_tableu0
        chi = len(states)
        k = len(measured_qubits)
        n = states[0][0].shape[0]

        y = np.array([state[2] for state in states])

        # Precompute original tableaus
        X_list = [s[0][:, 0::2] for s in states]
        Z_list = [s[0][:, 1::2] for s in states]
        p_list = [s[1] for s in states]

        total = 0.0 + 0.0j

        for s_bits in range(1 << k):

            z_mask = np.zeros(n, dtype=int)
            s_vec = np.zeros(k, dtype=int)

            for i in range(k):
                if (s_bits >> i) & 1:
                    z_mask[measured_qubits[i]] = 1
                    s_vec[i] = 1

            phase = (-1) ** (np.dot(s_vec, x) % 2)

            val = 0.0 + 0.0j

            for b in range(chi):

                # Apply Pauli to state b
                tab_b, p_b, y_b = states[b]
                tab_b2, p_b2 = self.apply_pauli_Z(tab_b,p_b,z_mask)

                Xb2 = tab_b2[:, 0::2]
                Zb2 = tab_b2[:, 1::2]

                inner_sum = 0.0 + 0.0j

                # Compute overlap with ALL a
                for a in range(chi):

                    overlap = self.stabilizer_inner_product_fast(
                        X_list[a], Z_list[a], p_list[a],
                        Xb2, Zb2, p_b2
                    )

                    inner_sum += np.conj(y[a]) * overlap

                val += y[b] * inner_sum

            total += phase * val

        numerator = total / (2 ** k)

        # normalization
        norm = 0.0 + 0.0j
        for a in range(chi):
            for b in range(chi):
                overlap = self.stabilizer_inner_product_fast(
                    X_list[a], Z_list[a], p_list[a],
                    X_list[b], Z_list[b], p_list[b]
                )
                norm += np.conj(y[a]) * y[b] * overlap

        return np.real(numerator / norm)

    def compute_probability_Approach_3(self, measured_qubits, x, num_samples=2000):

        states = self.global_tableu0
        chi = len(states)
        k = len(measured_qubits)
        n = states[0][0].shape[0]

        y = np.array([state[2] for state in states], dtype=complex)

        probs = np.abs(y)**2
        probs /= probs.sum()
        norm = 0.0 + 0.0j
        for a in range(chi):
            Xa = states[a][0][:, 0::2]
            Za = states[a][0][:, 1::2]
            pa = states[a][1]

            for b in range(chi):
                Xb = states[b][0][:, 0::2]
                Zb = states[b][0][:, 1::2]
                pb = states[b][1]

                overlap = self.stabilizer_inner_product_fast(Xa,Za,pa,Xb,Zb,pb)
                norm += np.conj(y[a]) * y[b] * overlap

        total = 0.0 + 0.0j

        for _ in range(num_samples):
            a = np.random.choice(chi, p=probs)
            b = np.random.choice(chi, p=probs)

            tab_a, p_a, y_a = states[a]
            tab_b, p_b, y_b = states[b]
            sample_val = 0.0 + 0.0j
            s_vec = np.random.randint(0, 2, size=k)

            z_mask = np.zeros(n, dtype=int)
            for i, q in enumerate(measured_qubits):
                if s_vec[i]:
                    z_mask[q] = 1

            phase = (-1) ** (np.dot(s_vec, x) % 2)
            tab_b2, p_b2 = self.apply_pauli_Z(tab_b, p_b, z_mask)

            Xb = tab_b2[:, 0::2]
            Zb = tab_b2[:, 1::2]

            Xa = tab_a[:, 0::2]
            Za = tab_a[:, 1::2]

            overlap = self.stabilizer_inner_product_fast(
                Xa, Za, p_a,
                Xb, Zb, p_b2
            )
            weight = (np.conj(y[a]) * y[b]) / (probs[a] * probs[b])

            total += phase * weight * overlap

        return max(np.real(total / num_samples / norm),0)
    
    def compute_marginal_Approach_2(self, qubit):

        states = self.global_tableu0
        chi = len(states)
        n = states[0][0].shape[0]

        y = np.array([state[2] for state in states])

        X_list = [s[0][:, 0::2] for s in states]
        Z_list = [s[0][:, 1::2] for s in states]
        p_list = [s[1] for s in states]
        z_mask = np.zeros(n, dtype=int)
        z_mask[qubit] = 1

        val = 0.0 + 0.0j

        for b in range(chi):

            tab_b, p_b, y_b = states[b]
            tab_b2, p_b2 = self.apply_pauli_Z(tab_b, p_b, z_mask)

            Xb2 = tab_b2[:, 0::2]
            Zb2 = tab_b2[:, 1::2]

            inner_sum = 0.0 + 0.0j

            for a in range(chi):

                overlap = self.stabilizer_inner_product_fast(
                    X_list[a], Z_list[a], p_list[a],
                    Xb2, Zb2, p_b2
                )

                inner_sum += np.conj(y[a]) * overlap

            val += y[b] * inner_sum

        # normalization
        norm = 0.0 + 0.0j
        for a in range(chi):
            for b in range(chi):
                overlap = self.stabilizer_inner_product_fast(
                    X_list[a], Z_list[a], p_list[a],
                    X_list[b], Z_list[b], p_list[b]
                )
                norm += np.conj(y[a]) * y[b] * overlap

        exp_Z = np.real(val / norm)

        return (1 - exp_Z) / 2
    
    def compute_marginal_Approach_3(self, qubit, num_samples=2000):

        states = self.global_tableu0
        chi = len(states)
        n = states[0][0].shape[0]

        y = np.array([state[2] for state in states], dtype=complex)

        probs = np.abs(y)**2
        probs /= probs.sum()

        # normalization
        norm = 0.0 + 0.0j
        for a in range(chi):
            Xa = states[a][0][:, 0::2]
            Za = states[a][0][:, 1::2]
            pa = states[a][1]

            for b in range(chi):
                Xb = states[b][0][:, 0::2]
                Zb = states[b][0][:, 1::2]
                pb = states[b][1]

                overlap = self.stabilizer_inner_product_fast(Xa,Za,pa,Xb,Zb,pb)
                norm += np.conj(y[a]) * y[b] * overlap

        total = 0.0 + 0.0j

        z_mask = np.zeros(n, dtype=int)
        z_mask[qubit] = 1

        for _ in range(num_samples):

            a = np.random.choice(chi, p=probs)
            b = np.random.choice(chi, p=probs)

            tab_a, p_a, y_a = states[a]
            tab_b, p_b, y_b = states[b]

            tab_b2, p_b2 = self.apply_pauli_Z(tab_b, p_b, z_mask)

            overlap = self.stabilizer_inner_product_fast(
                tab_a[:, 0::2], tab_a[:, 1::2], p_a,
                tab_b2[:, 0::2], tab_b2[:, 1::2], p_b2
            )

            weight = (np.conj(y[a]) * y[b]) / (probs[a] * probs[b])
            total += weight * overlap

        exp_Z = np.real(total / (num_samples * norm))

        return max((1 - exp_Z) / 2,0)