# STAVAN: Stabilizer Tableau framework for Verification of Nearly Clifford Quantum Circuits


**STAVAN** is a high-performance classical simulation and verification framework for universal quantum circuits based on the **Clifford + T** gate set. Our work builds upon the stabilizer formalism by Aaronson et al. [1] with the optimized magic state decomposition method established by Bravyi and Gosset [2],[3] to represent non-Clifford magic states as a linear combination of stabilizer states. STAVAN enables the verification of medium-scale quantum circuits with high deterministic fidelity.

## Key Features
* **Stabilizer tableau**: We implement a novel adaptation of the stabilizer tableau framework, with low memory footprint, to accommodate non-Clifford circuits.
* **Deterministic amplitude extraction**: By utilizing a Reduced Row Echelon Form (RREF) approach on the stabilizer tableau, we provide exact analytical amplitudes within the stabilizer subspace.
* **Gram matrix pre-computation**: We introduce a caching and pre-computation layer for the Gram matrix of the stabilizer decomposition. This reduces the complexity of the deterministic method, enabling the simulation of T -counts as high as t = 48 on standard classical hardware.
* **Probability Estimation**: We develop and evaluate three distinct approaches for computing output bit string probabilities:
    1. **Approach 1**: Exact Decomposition of Ancilla state into exponential number of orthogonal stabilizer states.
    2. **Approach 2** (Deterministic approach): Approaximating the Ancilla state to $\chi$ terms used for verification in low $T$-count circuits.
    3. **Approach 3** (Monte Carlo based sampling): A randomized approach that utilizes importance sampling and $N$ random stabilizer states to reduce the complexity to $O(N \cdot n^3)$. In STAVAN, we improve the time complexity from $O(\chi \cdot N \cdot poly(n))$ in existing methods to $O(\chi \cdot n^3 + N \cdot n^3 )$.

In Approach 3, STAVAN optimizes the time complexity and the memory footprint,  by implementing the stabilizer tableau and a pre-computed Gram Matrix.

| Task | Existing Methods | STAVAN Framework (Ours) |
| :--- | :--- | :--- |
| Estimating $P(x)$ | $O(\chi^2 \cdot poly(n))$ | $O(\chi^2 \cdot poly(n))$ |
| Sampling $x$ | $O(\chi \cdot N \cdot poly(n))$ | $O(\chi \cdot n^3 + N \cdot n^3)$ |
| Inner product calculation | $O(n^3)$ | $O(n^3)$ with much lesser memory footprint |

## Installation

```bash
# Clone the repository
git clone https://github.com/QCE26/STAVAN.git
cd STAVAN
```

## Repositiory Structure
```text
STAVAN/
├── STAVAN_Framework.py
├── demo.ipynb
├── testing/
│   ├── benchmarking.py
│   └── testing.py
├── LICENSE
└── README.md
```

- The Python based code for STAVAN framework is available in ```STAVAN_Framework.py``` file. MATLAB version would be made available soon.
- To initialize the Stabilizer Tableau, you need to provide the number of qubits $n$, $T$-count (if you use Approach 2 or 3) before hand. If you use Approach 3, you need to provide the number of samples $N$ when calculating output probability. ```demo.ipynb``` provides details regarding the same.
- ```benchmarking.py``` contains the Hard- $T$ structure based quantum circuits (from Clifford + T gate set) that were used for simulations and performative analysis. ```testing.py``` gives a template script of using the STAVAN Framework to simulate quantum circuits.

## Key References
[1]: S. Aaronson and D. Gottesman, “Improved simulation of stabilizer circuits,” Phys. Rev. A, vol. 70, p. 052328, Nov 2004.

[2] S. Bravyi and D. Gosset, “Improved classical simulation of quantum circuits dominated by clifford gates,” Phys. Rev. Lett., vol. 116, p. 250501, Jun 2016.

[3] S. Bravyi, D. Browne, P. Calpin, E. Campbell, D. Gosset, and M. Howard, “Simulation of quantum circuits by low-rank stabilizer decompositions,” Quantum, vol. 3, p. 181, Sep 2019.
## License
This project is licensed under the MIT License. See ```LICENSE``` file for details.
