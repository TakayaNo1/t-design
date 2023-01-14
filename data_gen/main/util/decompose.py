import numpy as np
from common import I,X,Y,Z, pauli_exp

"""Reference
QASM https://arxiv.org/abs/1707.03429
"""

def U3(t1 :float ,t2: float,t3 : float) -> np.ndarray:
    """construct U3 operator defined in QASM
    
    Arguments:
        t1 {float} -- theta (second Y rotation)
        t2 {float} -- phi (last Z rotation)
        t3 {float} -- lambda (first Z rotation)
    
    Returns:
        np.ndarray -- matrix representation of U3
    """

    return pauli_exp(Z,-t2)@pauli_exp(Y,-t1)@pauli_exp(Z,-t3)

def decompose_unitary_to_U3(u: np.ndarray) -> list:
    """decompose unitary to U3 defined in QASM
    
    Arguments:
        u {np.ndarray} -- 2*2 unitary matrix
    
    Returns:
        list -- three angles used in U3 gate
    """

    angle1 = np.angle(u[1,1])
    angle2 = np.angle(u[1,0])
    t2 = angle1+angle2
    t3 = angle1-angle2
    cv = u[1,1]/np.exp(1.j*angle1)
    sv = u[1,0]/np.exp(1.j*angle2)
    t1 = np.arccos(np.real(cv))*2
    if(sv<0):
        t1 = -t1
    return t1,t2,t3

def ZXZXZ_matrix(z1:float,z2:float,z3:float)->np.ndarray:
    """construct unitary matrix from three-Z rotation defined in QASM
    
    Arguments:
        z1 {float} -- last Z rotation
        z2 {float} -- second Z rotation
        z3 {float} -- first Z rotation
    
    Returns:
        np.ndarray -- matrix representation of ZXZXZ operation
    """

    return pauli_exp(Z,-z1)@pauli_exp(X,-np.pi/2)@pauli_exp(Z,-z2) \
         @ pauli_exp(X,-np.pi/2) @ pauli_exp(Z,-z3)


def construct_ZXZXZ_sequence(u: np.ndarray) -> list:
    """decompose unitary to Z-rot X-half-pi Z-rot X-half-pi Z-rot
    
    Arguments:
        u {np.ndarray} -- 2*2 unitary matrix
    
    Returns:
        list -- list of three Z rotation angle
    """

    t1,t2,t3 = decompose_unitary_to_U3(u)
    return t2 + 3*np.pi , t1+np.pi, t3

def test_decompose():
    """test function for decompositions
    """

    from icosahedral_group import IcosahedralGroup
    ig = IcosahedralGroup()
    for u in ig.sample(100):
        t1,t2,t3 = decompose_unitary_to_U3(u)
        ur1 = U3(t1,t2,t3)
        z1,z2,z3 = construct_ZXZXZ_sequence(u)
        ur2 = ZXZXZ_matrix(z1,z2,z3)
        assert(np.allclose(u,ur1))
        assert(np.allclose(u,ur2))

    from clifford_group import CliffordGroup
    cf = CliffordGroup(1)
    for u in cf.sample(100):
        t1,t2,t3 = decompose_unitary_to_U3(u)
        ur1 = U3(t1,t2,t3)
        z1,z2,z3 = construct_ZXZXZ_sequence(u)
        ur2 = ZXZXZ_matrix(z1,z2,z3)
        assert(np.allclose(u,ur1))
        assert(np.allclose(u,ur2))

if __name__ == "__main__":
    test_decompose()
