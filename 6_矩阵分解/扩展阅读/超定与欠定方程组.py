# 设置路径和导入utils
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from utils import myDisplay

# 导入必要的库
import numpy as np
import sympy as sp
from scipy.linalg import qr, svd, lstsq, pinv
from scipy.optimize import minimize

# 设置sympy的显示方式为latex
sp.init_printing(use_latex=True)

def solve_overdetermined():
    """
    解决超定方程组的示例
    使用不同方法求解 Ax ≈ b
    """
    # 创建一个超定方程组
    A = sp.Matrix([[1, 1],
                   [2, 1],
                   [1, 2]])
    b = sp.Matrix([2, 3, 3])
    
    print("超定方程组示例：")
    print("A =")
    myDisplay(A)
    print("b =")
    myDisplay(b)
    
    # 转换为numpy数组
    A_np = np.array(A).astype(float)
    b_np = np.array(b).astype(float).reshape(-1)
    
    # 1. 使用QR分解求解
    print("\n1. QR分解方法:")
    Q, R = qr(A_np, mode='economic')
    x_qr = np.linalg.solve(R, Q.T @ b_np)
    myDisplay(sp.Matrix(x_qr))
    
    # 2. 使用SVD分解求解
    print("\n2. SVD方法:")
    U, s, VT = svd(A_np, full_matrices=False)
    x_svd = VT.T @ (U.T @ b_np / s)
    myDisplay(sp.Matrix(x_svd))
    
    # 3. 使用正规方程求解
    print("\n3. 正规方程方法:")
    x_normal = np.linalg.solve(A_np.T @ A_np, A_np.T @ b_np)
    myDisplay(sp.Matrix(x_normal))
    
    # 计算残差
    print("\n各方法的残差 ||Ax - b||:")
    print(f"QR方法: {np.linalg.norm(A_np @ x_qr - b_np)}")
    print(f"SVD方法: {np.linalg.norm(A_np @ x_svd - b_np)}")
    print(f"正规方程: {np.linalg.norm(A_np @ x_normal - b_np)}")

def solve_underdetermined():
    """
    解决欠定方程组的示例
    使用不同方法求解 Ax = b
    """
    # 创建一个欠定方程组
    A = sp.Matrix([[1, 2, 1],
                   [2, 1, 2]])
    b = sp.Matrix([4, 5])
    
    print("\n欠定方程组示例：")
    print("A =")
    myDisplay(A)
    print("b =")
    myDisplay(b)
    
    # 转换为numpy数组
    A_np = np.array(A).astype(float)
    b_np = np.array(b).astype(float).reshape(-1)
    
    # 1. 使用SVD求最小范数解
    print("\n1. SVD最小范数解:")
    # 使用伪逆求解
    x_svd = pinv(A_np) @ b_np
    myDisplay(sp.Matrix(x_svd))
    
    # 2. L2正则化（岭回归）
    print("\n2. L2正则化解:")
    lambda_l2 = 0.1
    x_l2 = np.linalg.solve(A_np.T @ A_np + lambda_l2 * np.eye(3), A_np.T @ b_np)
    myDisplay(sp.Matrix(x_l2))
    
    # 3. L1正则化（LASSO）
    print("\n3. L1正则化解:")
    def lasso_objective(x):
        return np.linalg.norm(A_np @ x - b_np)**2 + 0.1 * np.sum(np.abs(x))
    
    x0 = np.zeros(3)
    result = minimize(lasso_objective, x0, method='BFGS')
    x_l1 = result.x
    myDisplay(sp.Matrix(x_l1))
    
    # 比较不同解的范数
    print("\n各解的范数:")
    print(f"SVD最小范数解: {np.linalg.norm(x_svd)}")
    print(f"L2正则化解: {np.linalg.norm(x_l2)}")
    print(f"L1正则化解: {np.linalg.norm(x_l1)}")
    
    # 验证是否满足原方程
    print("\n验证方程 Ax = b 的残差:")
    print(f"SVD解残差: {np.linalg.norm(A_np @ x_svd - b_np)}")
    print(f"L2解残差: {np.linalg.norm(A_np @ x_l2 - b_np)}")
    print(f"L1解残差: {np.linalg.norm(A_np @ x_l1 - b_np)}")

def practical_example():
    """
    实际应用示例：数据拟合
    """
    # 生成带噪声的数据点
    x_data = np.linspace(0, 5, 20)
    y_true = 2 * x_data + 1
    y_data = y_true + np.random.normal(0, 0.5, size=len(x_data))
    
    # 构建超定方程组
    A = np.vstack([x_data, np.ones_like(x_data)]).T
    
    print("\n实际应用：线性回归")
    print("使用最小二乘法拟合 y = ax + b")
    
    # 使用不同方法求解
    # 1. QR分解
    Q, R = qr(A, mode='economic')
    x_qr = np.linalg.solve(R, Q.T @ y_data)
    print("\nQR分解结果:")
    print(f"y = {x_qr[0]:.3f}x + {x_qr[1]:.3f}")
    
    # 2. SVD分解
    U, s, VT = svd(A, full_matrices=False)
    x_svd = VT.T @ (U.T @ y_data / s)
    print("\nSVD分解结果:")
    print(f"y = {x_svd[0]:.3f}x + {x_svd[1]:.3f}")

if __name__ == "__main__":
    print("=== 超定方程组求解 ===")
    solve_overdetermined()
    
    print("\n=== 欠定方程组求解 ===")
    solve_underdetermined()
    
    print("\n=== 实际应用示例 ===")
    practical_example() 