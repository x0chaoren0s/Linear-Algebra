# 设置路径和导入utils
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from utils import myDisplay

# 导入其他必要的库
import numpy as np
import sympy as sp
from scipy.linalg import qr
import numpy.linalg as LA

# 设置sympy的显示方式为latex
sp.init_printing(use_latex=True)

def gram_schmidt_qr(A):
    """
    使用Gram-Schmidt正交化实现QR分解
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = LA.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

def householder_qr(A):
    """
    使用Householder变换实现QR分解
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for j in range(n):
        x = R[j:, j]
        e = np.zeros_like(x)
        e[0] = LA.norm(x)
        u = x - e
        v = u / LA.norm(u) if LA.norm(u) > 0 else u
        Q2 = np.eye(m)
        Q2[j:, j:] -= 2.0 * np.outer(v, v)
        R = Q2 @ R
        Q = Q @ Q2.T
        
    return Q, R

def givens_qr(A):
    """
    使用Givens旋转实现QR分解
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for j in range(n):
        for i in range(m-1, j, -1):
            # 构造Givens旋转矩阵
            c = R[i-1, j] / np.sqrt(R[i-1, j]**2 + R[i, j]**2)
            s = -R[i, j] / np.sqrt(R[i-1, j]**2 + R[i, j]**2)
            G = np.eye(m)
            G[i-1:i+1, i-1:i+1] = np.array([[c, -s], [s, c]])
            R = G @ R
            Q = Q @ G.T
            
    return Q, R

def compare_methods():
    """
    比较不同QR分解方法
    """
    # 创建测试矩阵
    A = sp.Matrix([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    print("原始矩阵 A:")
    myDisplay(A)
    
    A_np = np.array(A).astype(float)
    
    # 1. 使用Gram-Schmidt方法
    print("\n1. Gram-Schmidt QR分解:")
    Q_gs, R_gs = gram_schmidt_qr(A_np)
    myDisplay(sp.Matrix(Q_gs))
    myDisplay(sp.Matrix(R_gs))
    
    # 2. 使用Householder方法
    print("\n2. Householder QR分解:")
    Q_hh, R_hh = householder_qr(A_np)
    myDisplay(sp.Matrix(Q_hh))
    myDisplay(sp.Matrix(R_hh))
    
    # 3. 使用Givens方法
    print("\n3. Givens QR分解:")
    Q_gv, R_gv = givens_qr(A_np)
    myDisplay(sp.Matrix(Q_gv))
    myDisplay(sp.Matrix(R_gv))
    
    # 4. 使用scipy的QR分解（作为参考）
    print("\n4. SciPy QR分解:")
    Q_sp, R_sp = qr(A_np)
    myDisplay(sp.Matrix(Q_sp))
    myDisplay(sp.Matrix(R_sp))

def numerical_stability_test():
    """
    测试不同方法的数值稳定性
    """
    # 创建一个病态矩阵
    n = 5
    A = np.vander(np.linspace(1, 2, n))  # Vandermonde矩阵通常是病态的
    print("\n病态矩阵 A:")
    myDisplay(sp.Matrix(A))
    
    # 比较不同方法的正交性误差
    print("\n正交性误差 ||Q^T Q - I||_F:")
    
    # Gram-Schmidt
    Q_gs, _ = gram_schmidt_qr(A)
    err_gs = LA.norm(Q_gs.T @ Q_gs - np.eye(n), 'fro')
    print(f"Gram-Schmidt: {err_gs}")
    
    # Householder
    Q_hh, _ = householder_qr(A)
    err_hh = LA.norm(Q_hh.T @ Q_hh - np.eye(n), 'fro')
    print(f"Householder: {err_hh}")
    
    # Givens
    Q_gv, _ = givens_qr(A)
    err_gv = LA.norm(Q_gv.T @ Q_gv - np.eye(n), 'fro')
    print(f"Givens: {err_gv}")
    
    # SciPy
    Q_sp, _ = qr(A)
    err_sp = LA.norm(Q_sp.T @ Q_sp - np.eye(n), 'fro')
    print(f"SciPy: {err_sp}")

if __name__ == "__main__":
    print("=== QR分解方法比较 ===")
    compare_methods()
    
    print("\n=== 数值稳定性测试 ===")
    numerical_stability_test() 