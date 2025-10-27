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

# Dự báo tại (X1=25, X2=9) để khớp với file
x_new = np.array([1.0, 25.0, 9.0])
y_hat_new = float(x_new @ b)
print(f"Dự báo tại (X1=25, X2=9): Ŷ = {y_hat_new:.4f}")

# -----------------------------
# VẼ HÌNH
# Lưu ý yêu cầu: mỗi chart 1 figure, không set màu cụ thể.
# -----------------------------
outdir = Path("/mnt/data")
outdir.mkdir(parents=True, exist_ok=True)

# (1) 3D scatter + fitted plane
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X1, X2, Y, label="Dữ liệu")

# Tạo lưới để vẽ mặt phẳng
x1g = np.linspace(X1.min(), X1.max(), 20)
x2g = np.linspace(X2.min(), X2.max(), 20)
X1g, X2g = np.meshgrid(x1g, x2g)
Yg = b0 + b1*X1g + b2*X2g

ax.plot_surface(X1g, X2g, Yg, alpha=0.3)
ax.set_xlabel("X1 (Quảng cáo)")
ax.set_ylabel("X2 (Nhân viên)")
ax.set_zlabel("Y (Doanh thu)")
ax.set_title("Hồi quy đa biến: mặt phẳng fitted")
ax.legend()
fig1.savefig(outdir / "multivar_reg_plane_3d.png", bbox_inches="tight", dpi=150)

# (2) Projection: Y vs X1 (giữ X2 = mean)
fig2 = plt.figure()
x2_mean = X2.mean()
x1_line = np.linspace(X1.min(), X1.max(), 100)
y_line1 = b0 + b1*x1_line + b2*x2_mean
plt.scatter(X1, Y, label="Dữ liệu (X2 thay đổi)")
plt.plot(x1_line, y_line1, label=f"X2 cố định = {x2_mean:.2f}")
plt.xlabel("X1 (Quảng cáo)")
plt.ylabel("Y (Doanh thu)")
plt.title("Dự báo Y theo X1 (X2 cố định tại mean)")
plt.legend()
fig2.savefig(outdir / "multivar_reg_y_vs_x1.png", bbox_inches="tight", dpi=150)

# (3) Projection: Y vs X2 (giữ X1 = mean)
fig3 = plt.figure()
x1_mean = X1.mean()
x2_line = np.linspace(X2.min(), X2.max(), 100)
y_line2 = b0 + b1*x1_mean + b2*x2_line
plt.scatter(X2, Y, label="Dữ liệu (X1 thay đổi)")
plt.plot(x2_line, y_line2, label=f"X1 cố định = {x1_mean:.2f}")
plt.xlabel("X2 (Nhân viên)")
plt.ylabel("Y (Doanh thu)")
plt.title("Dự báo Y theo X2 (X1 cố định tại mean)")
plt.legend()
fig3.savefig(outdir / "multivar_reg_y_vs_x2.png", bbox_inches="tight", dpi=150)

print("Đã lưu hình:")
print(" - multivar_reg_plane_3d.png")
print(" - multivar_reg_y_vs_x1.png")
print(" - multivar_reg_y_vs_x2.png")


# -----------------------------
# VẼ HISTOGRAM
# -----------------------------
Y_hat = b0 + b1*X1 + b2*X2
resid = Y - Y_hat

# Histogram Y thực tế và Y dự báo
plt.figure()
plt.hist(Y, bins=5, alpha=0.5, label="Y thực tế")
plt.hist(Y_hat, bins=5, alpha=0.5, label="Y dự báo")
plt.xlabel("Giá trị")
plt.ylabel("Tần suất")
plt.title("Histogram Y thực tế và Y dự báo")
plt.legend()
plt.show()

# Histogram phần dư
plt.figure()
plt.hist(resid, bins=5, alpha=0.7, edgecolor='black')
plt.xlabel("Giá trị phần dư")
plt.ylabel("Tần suất")
plt.title("Histogram phần dư (Residuals)")
plt.show()
