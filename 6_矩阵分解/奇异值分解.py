# 设置路径和导入utils
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils import myDisplay

# 导入其他必要的库
import numpy as np
import sympy as sp
from scipy.linalg import svd, eig
import matplotlib.pyplot as plt

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置sympy的显示方式为latex
sp.init_printing(use_latex=True)

def demonstrate_basic_svd():
    """基本SVD分解演示"""
    # 示例矩阵
    A = sp.Matrix([[4, 0],
                   [3, -5]])
    print("原始矩阵 A:")
    myDisplay(A)

    # SVD分解
    A_np = np.array(A).astype(float)
    U, Sigma, VT = svd(A_np)

    # 转换为sympy矩阵，注意Sigma需要正确构造为对角矩阵
    U_sp = sp.Matrix(U)
    # 创建合适大小的对角矩阵
    Sigma_sp = sp.Matrix.zeros(A.shape[0], A.shape[1])
    for i in range(len(Sigma)):
        Sigma_sp[i, i] = Sigma[i]
    VT_sp = sp.Matrix(VT)

    print("\n左奇异向量矩阵 U:")
    myDisplay(U_sp)
    print("U的正交性验证 (U^T × U = I):")
    myDisplay(U_sp.T * U_sp)

    print("\n奇异值矩阵 Σ:")
    myDisplay(Sigma_sp)
    print("奇异值：", Sigma)

    print("\n右奇异向量矩阵 V^T:")
    myDisplay(VT_sp)
    print("V的正交性验证 (V × V^T = I):")
    myDisplay(VT_sp.T * VT_sp)

    print("\n验证 A = U × Σ × V^T:")
    myDisplay(U_sp * Sigma_sp * VT_sp)

def verify_properties():
    """验证SVD的重要性质"""
    A = sp.Matrix([[4, 0],
                   [3, -5]])
    A_np = np.array(A).astype(float)
    U, Sigma, VT = svd(A_np)
    
    # 1. 验证奇异值的平方是A^T A的特征值
    print("\n验证性质：")
    print("1. 奇异值的平方是A^T A的特征值")
    ATA = A_np.T @ A_np
    eigenvals = np.linalg.eigvals(ATA)
    print("A^T A的特征值:", np.sort(eigenvals)[::-1])
    print("奇异值的平方:", Sigma**2)
    
    # 2. 验证U的列是AA^T的特征向量
    print("\n2. U的列是AA^T的特征向量")
    AAT = A_np @ A_np.T
    AAT_eigvals, AAT_eigvecs = eig(AAT)
    print("AA^T的特征向量:")
    myDisplay(sp.Matrix(AAT_eigvecs))
    print("U矩阵:")
    myDisplay(sp.Matrix(U))

def demonstrate_geometric_transformation():
    """演示SVD的几何变换过程"""
    A = sp.Matrix([[4, 0],
                   [3, -5]])
    A_np = np.array(A).astype(float)
    U, Sigma, VT = svd(A_np)
    
    # 创建单位圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # 1. V^T的作用
    step1 = VT @ circle
    # 2. Sigma的作用
    step2 = np.diag(Sigma) @ step1
    # 3. U的作用
    step3 = U @ step2
    
    # 绘制变换过程
    plt.figure(figsize=(15, 5))
    
    # 原始圆
    plt.subplot(141)
    plt.plot(circle[0], circle[1])
    plt.title('原始单位圆')
    plt.axis('equal')
    plt.grid(True)
    
    # V^T变换后
    plt.subplot(142)
    plt.plot(step1[0], step1[1])
    plt.title('V^T旋转后')
    plt.axis('equal')
    plt.grid(True)
    
    # Sigma变换后
    plt.subplot(143)
    plt.plot(step2[0], step2[1])
    plt.title('Σ拉伸后')
    plt.axis('equal')
    plt.grid(True)
    
    # U变换后
    plt.subplot(144)
    plt.plot(step3[0], step3[1])
    plt.title('U旋转后（最终结果）')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_matrix_approximation():
    """演示矩阵近似"""
    # 创建一个图像矩阵（简单的例子）
    A = np.zeros((10, 10))
    A[2:8, 2:8] = 1  # 创建一个方块
    
    # SVD分解
    U, Sigma, VT = svd(A)
    
    # 使用不同数量的奇异值重构
    ranks = [1, 2, 5]
    plt.figure(figsize=(15, 3))
    
    # 原始图像
    plt.subplot(141)
    plt.imshow(A, cmap='gray')
    plt.title('原始图像')
    
    # 不同秩的近似
    for i, r in enumerate(ranks, 1):
        A_approx = U[:, :r] @ np.diag(Sigma[:r]) @ VT[:r, :]
        plt.subplot(141 + i)
        plt.imshow(A_approx, cmap='gray')
        plt.title(f'秩{r}近似')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== SVD基本分解演示 ===")
    demonstrate_basic_svd()
    
    print("\n=== SVD性质验证 ===")
    verify_properties()
    
    print("\n=== SVD几何变换演示 ===")
    demonstrate_geometric_transformation()
    
    print("\n=== SVD矩阵近似演示 ===")
    demonstrate_matrix_approximation()