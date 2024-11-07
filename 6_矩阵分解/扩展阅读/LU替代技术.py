# 设置路径和导入utils
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from utils import myDisplay

# 导入其他必要的库
import numpy as np
import sympy as sp
from scipy.linalg import lu, cholesky, qr

# 设置sympy的显示方式为latex
sp.init_printing(use_latex=True)

def compare_decompositions():
    # 创建一个对称正定矩阵用于测试
    A = sp.Matrix([[4, 2, 0],
                   [2, 5, 2],
                   [0, 2, 4]])
    print("原始矩阵 A:")
    myDisplay(A)
    
    # 转换为numpy数组
    A_np = np.array(A).astype(float)
    
    # 1. LU分解
    print("\n1. LU分解:")
    P, L, U = lu(A_np)
    myDisplay(sp.Matrix(L))
    myDisplay(sp.Matrix(U))
    
    # 2. Cholesky分解
    print("\n2. Cholesky分解:")
    L_chol = cholesky(A_np, lower=True)
    L_chol_sp = sp.Matrix(L_chol)
    myDisplay(L_chol_sp)
    print("验证 L × L^T:")
    myDisplay(L_chol_sp * L_chol_sp.T)
    
    # 3. QR分解
    print("\n3. QR分解:")
    Q, R = qr(A_np)
    Q_sp = sp.Matrix(Q)
    R_sp = sp.Matrix(R)
    myDisplay(Q_sp)
    myDisplay(R_sp)

def solve_system_comparison():
    """比较不同分解方法求解线性方程组的效果"""
    # 创建测试矩阵和向量
    A = sp.Matrix([[4, 2, 0],
                   [2, 5, 2],
                   [0, 2, 4]])
    b = sp.Matrix([1, 2, 3])
    print("\n求解方程 Ax = b")
    print("A =")
    myDisplay(A)
    print("b =")
    myDisplay(b)
    
    # 转换为numpy数组
    A_np = np.array(A).astype(float)
    b_np = np.array(b).astype(float)
    
    # 1. 使用LU分解求解
    print("\n1. 使用LU分解求解:")
    P, L, U = lu(A_np)
    y = np.linalg.solve(L, P @ b_np)
    x_lu = np.linalg.solve(U, y)
    myDisplay(sp.Matrix(x_lu))
    
    # 2. 使用Cholesky分解求解
    print("\n2. 使用Cholesky分解求解:")
    L_chol = cholesky(A_np, lower=True)
    y = np.linalg.solve(L_chol, b_np)
    x_chol = np.linalg.solve(L_chol.T, y)
    myDisplay(sp.Matrix(x_chol))
    
    # 3. 使用QR分解求解
    print("\n3. 使用QR分解求解:")
    Q, R = qr(A_np)
    x_qr = np.linalg.solve(R, Q.T @ b_np)
    myDisplay(sp.Matrix(x_qr))

if __name__ == "__main__":
    print("=== 不同分解方法的比较 ===")
    compare_decompositions()
    
    print("\n=== 求解线性方程组的比较 ===")
    solve_system_comparison() 