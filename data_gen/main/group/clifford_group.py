import numpy as np
from .group_base import GroupBase
from util.common import I,X,Y,Z,H,S,CZ, list_product

"""Reference
Unique decomposition https://arxiv.org/abs/1310.6813
"""

class CliffordGroup(GroupBase):
    def __init__(self, num_qubit : int) -> None:
        """Constructor of CliffordGroup class
        
        Arguments:
            num_qubit {int} -- number of qubits

        """
        self.num_qubit = num_qubit
        self.name = "Clifford"

        IH = np.kron(I,H)
        HI = np.kron(H,I)
        HH = np.kron(H,H)
        SI = np.kron(S,I)
        IS = np.kron(I,S)

        A1 = I
        A2 = H
        A3 = H@S@H
        B1 = IH@CZ@HH@CZ@HH@CZ
        B2 = CZ@HH@CZ
        B3 = HI@SI@CZ@HH@CZ
        B4 = HI@CZ@HH@CZ
        C1 = I
        C2 = H@S@S@H
        D1 = CZ@HH@CZ@HH@CZ@IH
        D2 = HI@CZ@HH@CZ@IH
        D3 = HH@IS@CZ@HH@CZ@IH
        D4 = HH@CZ@HH@CZ@IH
        E1 = I
        E2 = S
        E3 = S@S
        E4 = S@S@S

        self.A = [A1,A2,A3]
        self.B = [B1,B2,B3,B4]
        self.C = [C1,C2]
        self.D = [D1,D2,D3,D4]
        self.E = [E1,E2,E3,E4]

        self.Lcounts = [0]
        for ind in range(num_qubit):
            if ind == 0:
                self.Lcounts.append(len(self.C)*len(self.A))
            else:
                self.Lcounts.append(self.Lcounts[-1]*len(self.B))

        self.Rcounts = [0]
        for ind in range(num_qubit):
            if ind == 0:
                self.Rcounts.append(len(self.E))
            else:
                self.Rcounts.append(self.Rcounts[-1]*len(self.D))

        self.Lsum = [0]
        for ind in range(num_qubit):
            self.Lsum.append(self.Lsum[-1]+self.Lcounts[ind+1])

        self.order = 1
        for ind in range(1,num_qubit+1):
            self.order*=(2*(4**ind-1)*(4**ind))


    def get_element(self,index):
        assert(0 <= index and index < self.order)

        def pickM(n,index):
            mat = np.eye(2**n)
            for i in range(n-1):
                d = self.D[index%len(self.D)]
                index = index // len(self.D)
                d = np.kron(np.kron(np.eye(2**i),d),np.eye(2**(n-2-i)))
                mat = mat@d
            
            e = self.E[index%len(self.E)]
            e = np.kron(np.eye(2**(n-1)),e)
            mat = mat@e
            return mat

        def pickLm(n,m,index):
            assert(m<=n)
            mat = np.eye(2**n)
            a = self.A[index%len(self.A)]
            index = index // len(self.A)
            a = np.kron(np.kron(np.eye(2**(m-1)), a),np.eye(2**(n-m)))
            mat = mat@a

            for j in range(m-1):
                b = self.B[index%len(self.B)]
                index = index // len(self.B)
                b = np.kron(np.kron(np.eye(2**(m-2-j)),b),np.eye(2**(n-m+j)))
                mat = mat@b

            c = self.C[index%len(self.C)]
            c = np.kron(c,np.eye(2**(n-1)))
            mat = mat@c
            return mat

        def pickL(n,index):
            assert(index < self.Lsum[-1])
            for m in range(1,n+1):
                if index < self.Lsum[m]:
                    return pickLm(n,m,index-self.Lsum[m-1])

        matrix = np.eye(2**self.num_qubit)
        cnt = 1
        for n in range(self.num_qubit,0,-1):
            l = pickL(n,index%self.Lsum[n])
            index = index // self.Lsum[n]
            cnt *= self.Lsum[n]
            matrix = matrix @ np.kron(l,np.eye(2**(self.num_qubit-n)))

            m = pickM(n,index%self.Rcounts[n])
            index = index // self.Rcounts[n]
            cnt *= self.Rcounts[n]
            matrix = matrix @ np.kron(m,np.eye(2**(self.num_qubit-n)))
        assert(cnt==self.order)

        # apply global phase to convert SU
        det = np.linalg.det(matrix)
        matrix /= det**(2**(-self.num_qubit))
        #assert(abs(np.linalg.det(matrix)-1) < 1e-14)
        return matrix
    
