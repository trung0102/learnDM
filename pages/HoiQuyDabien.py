import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# -----------------------------
# DỮ LIỆU THEO ĐỀ TRONG FILE
# -----------------------------
stores = ["A", "B", "C", "D", "E"]
X1 = np.array([10, 20, 30, 40, 50], dtype=float)   # Quảng cáo
X2 = np.array([ 5,  7,  6,  8, 10], dtype=float)   # Nhân viên
Y  = np.array([40, 65, 70, 85, 100], dtype=float)  # Doanh thu

# -----------------------------
# CÔNG THỨC MA TRẬN
# -----------------------------
X = np.column_stack([np.ones_like(X1), X1, X2])  # (n, 3)
XtX = X.T @ X
XtY = X.T @ Y
XtX_inv = np.linalg.inv(XtX)
b = XtX_inv @ XtY   # [b0, b1, b2]

b0, b1, b2 = b
print("X^T X =\n", XtX)
print("\nX^T Y =\n", XtY)
print("\n(X^T X)^(-1) =\n", XtX_inv)
print("\nb = (X^T X)^(-1) X^T Y =\n", b)
print(f"\nPhương trình: Ŷ = {b0:.4f} + {b1:.4f}*X1 + {b2:.4f}*X2")