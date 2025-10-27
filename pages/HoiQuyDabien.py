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