# 设置路径和导入utils
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils import myDisplay

# 导入其他必要的库
import numpy as np
import sympy as sp
from scipy.linalg import lu

# 设置sympy的显示方式为latex
sp.init_printing(use_latex=True)

# 示例矩阵
A = sp.Matrix([[2, 3],
               [5, 4]])
print("原始矩阵 A:")
myDisplay(A)

# LU分解
A_np = np.array(A).astype(float)
P, L, U = lu(A_np)
P_sp = sp.Matrix(P)
L_sp = sp.Matrix(L)
U_sp = sp.Matrix(U)

print("\n置换矩阵 P:")
myDisplay(P_sp)

print("\n下三角矩阵 L:")
myDisplay(L_sp)

print("\n上三角矩阵 U:")
myDisplay(U_sp)

print("\n验证 P × A = L × U:")
myDisplay(P_sp * A)
print("等于")
myDisplay(L_sp * U_sp)

# 如果需要不带置换的LU分解，可以这样做：
print("\n还原原始顺序:")
myDisplay(P_sp.T * L_sp * U_sp)