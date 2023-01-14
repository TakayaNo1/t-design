import numpy as np
from .group_base import GroupBase
from util.common import I, X, Y, Z, H, S, CZ, list_product
import qulacs

"""Reference
Unique decomposition https://arxiv.org/abs/1310.6813
"""


class CliffordCircuitGroup(GroupBase):
    def __init__(self, num_qubit: int) -> None:
        """Constructor of CliffordGroup class

        Args:
            num_qubit (int): number of qubits
        """
        self.num_qubit = num_qubit
        self.name = "Clifford"

        IH = np.kron(I, H)
        HI = np.kron(H, I)
        HH = np.kron(H, H)
        SI = np.kron(S, I)
        IS = np.kron(I, S)

        A1 = I
        A2 = H
        A3 = H @ S @ H
        B1 = IH @ CZ @ HH @ CZ @ HH @ CZ
        B2 = CZ @ HH @ CZ
        B3 = HI @ SI @ CZ @ HH @ CZ
        B4 = HI @ CZ @ HH @ CZ
        C1 = I
        C2 = H @ S @ S @ H
        D1 = CZ @ HH @ CZ @ HH @ CZ @ IH
        D2 = HI @ CZ @ HH @ CZ @ IH
        D3 = HH @ IS @ CZ @ HH @ CZ @ IH
        D4 = HH @ CZ @ HH @ CZ @ IH
        E1 = I
        E2 = S
        E3 = S @ S
        E4 = S @ S @ S

        self.A = [A1, A2, A3]
        self.B = [B1, B2, B3, B4]
        self.C = [C1, C2]
        self.D = [D1, D2, D3, D4]
        self.E = [E1, E2, E3, E4]

        self.Lcounts = [0]
        for ind in range(num_qubit):
            if ind == 0:
                self.Lcounts.append(len(self.C) * len(self.A))
            else:
                self.Lcounts.append(self.Lcounts[-1] * len(self.B))

        self.Rcounts = [0]
        for ind in range(num_qubit):
            if ind == 0:
                self.Rcounts.append(len(self.E))
            else:
                self.Rcounts.append(self.Rcounts[-1] * len(self.D))

        self.Lsum = [0]
        for ind in range(num_qubit):
            self.Lsum.append(self.Lsum[-1] + self.Lcounts[ind + 1])

        self.order = 1
        for ind in range(1, num_qubit + 1):
            self.order *= (2 * (4**ind - 1) * (4**ind))

    def get_element(self, index: int) -> np.ndarray:
        """Get eleemnt of given index

        Index of Clifford may be much larger than 2^32, which is the limit of an default integer type.
        Thus, use python's default int, and don't use numpy's integer for large group.

        Args:
            index (int): index

        Returns:
            np.ndarray: element
        """
        assert(0 <= index and index < self.order)

        def pickM(n, index):
            sub_list = []
            for i in range(n - 1):
                rind = index % len(self.D)
                d = self.D[index % len(self.D)]
                index = index // len(self.D)
                d = (i, f"D{i}", rind, d)
                sub_list.append(d)

            rind = index % len(self.E)
            e = self.E[index % len(self.E)]
            e = (n - 1, "E", rind, e)
            sub_list.append(e)
            return sub_list

        def pickLm(n, m, index):
            assert(m <= n)
            sub_list = []

            rind = index % len(self.A)
            a = self.A[index % len(self.A)]
            index = index // len(self.A)
            a = (m - 1, "A", rind, a)
            sub_list.append(a)

            for j in range(m - 1):
                rind = index % len(self.B)
                b = self.B[index % len(self.B)]
                index = index // len(self.B)
                b = (m - 2 - j, f"B{j}", rind, b)
                sub_list.append(b)

            rind = index % len(self.C)
            c = self.C[index % len(self.C)]
            c = (0, "C", rind, c)
            sub_list.append(c)
            return sub_list

        def pickL(n, index):
            assert(index < self.Lsum[-1])
            for m in range(1, n + 1):
                if index < self.Lsum[m]:
                    return pickLm(n, m, index - self.Lsum[m - 1])

        gate_list = []
        cnt = 1
        for n in range(self.num_qubit, 0, -1):
            l = pickL(n, index % self.Lsum[n])
            index = index // self.Lsum[n]
            cnt *= self.Lsum[n]
            gate_list.extend(l)

            m = pickM(n, index % self.Rcounts[n])
            index = index // self.Rcounts[n]
            cnt *= self.Rcounts[n]
            gate_list.extend(m)
        assert(cnt == self.order)

        # apply global phase to convert SU
        return gate_list

    def _to_special_unitary(self, matrix):
        det = np.linalg.det(matrix)
        matrix /= det**(1 / matrix.shape[0])
        # assert(abs(np.linalg.det(matrix) - 1) < 1e-14)
        return matrix

    def _get_matrix(self, num_qubit, qind, mat):
        num_gate_qubit = int(np.round(np.log2(mat.shape[0])))
        pdim = 2**qind
        pmat = np.eye(pdim, dtype=complex)
        adim = 2**(num_qubit - qind - num_gate_qubit)
        amat = np.eye(adim, dtype=complex)
        return np.kron(np.kron(pmat, mat), amat)

    def circuit_to_matrix(self, num_qubit, circuit):
        result = np.eye(2**num_qubit, dtype=complex)
        for item in circuit:
            qind, _, _, mat = item
            result = result @ self._get_matrix(num_qubit, qind, mat)
        result = self._to_special_unitary(result)
        return result
    """
    def simulate_circuit(self, circuit, state):
        import qulacs
        for qind, _, _, mat in circuit:
            if mat.shape[0] == 2:
                gate = qulacs.gate.DenseMatrix(qind, mat)
            else:
                gate = qulacs.gate.DenseMatrix([qind, qind + 1], mat)
            gate.update_quantum_state(state)

    def simulate_circuit_specific_qubit(self, circuit, state, x):
        for qind, _, _, mat in circuit:
            if mat.shape[0] == 2:
                gate = qulacs.gate.DenseMatrix(qind + x, mat)
            else:
                gate = qulacs.gate.DenseMatrix([qind + x, qind + 1 + x], mat)
            gate.update_quantum_state(state)
    """
    def simulate_circuit(self, num_qubit, circuit, state):
        for qind, _, _, mat in reversed(circuit):
            if mat.shape[0] == 2:
                gate = qulacs.gate.DenseMatrix(num_qubit - 1 - qind, mat)
            else:
                gate = qulacs.gate.DenseMatrix([num_qubit - 2 - qind, num_qubit - 1 - qind], mat)
            gate.update_quantum_state(state)

    def simulate_circuit_specific_qubit(self, num_qubit, circuit, state, x):
        for qind, _, _, mat in reversed(circuit):
            if mat.shape[0] == 2:
                gate = qulacs.gate.DenseMatrix(num_qubit - 1 - qind + x, mat)
            else:
                gate = qulacs.gate.DenseMatrix([num_qubit - 2 - qind + x, num_qubit - 1 - qind + x], mat)
            gate.update_quantum_state(state)
            
