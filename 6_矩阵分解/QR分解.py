# 设置路径和导入utils
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils import myDisplay

# 导入其他必要的库
import numpy as np
import sympy as sp
from scipy.linalg import qr

# 设置sympy的显示方式为latex
sp.init_printing(use_latex=True)

# 示例矩阵
A = sp.Matrix([[1, 2],
               [3, 4]])
print("原始矩阵 A:")
myDisplay(A)

# QR分解
A_np = np.array(A).astype(float)
Q, R = qr(A_np)
Q_sp = sp.Matrix(Q)
R_sp = sp.Matrix(R)

print("\n正交矩阵 Q:")
myDisplay(Q_sp)

print("\n上三角矩阵 R:")
myDisplay(R_sp)

print("\n验证 Q^T × Q = I:")
myDisplay(Q_sp.T * Q_sp)

print("\n验证 Q × R:")
myDisplay(Q_sp * R_sp) 