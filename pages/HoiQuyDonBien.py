import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

# =============================
# HÀM TIỆN ÍCH HỒI QUY ĐƠN BIẾN
# =============================
def simple_linreg(x, y):
    """
    Hồi quy tuyến tính đơn biến y = b0 + b1*x
    Trả về dict chứa b0, b1, y_hat, residuals, SSE, SST, R2, SE(b0), SE(b1), df, s^2...
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    x_bar = x.mean()
    y_bar = y.mean()

    Sxx = np.sum((x - x_bar)**2)
    Sxy = np.sum((x - x_bar)*(y - y_bar))

    b1 = Sxy / Sxx
    b0 = y_bar - b1 * x_bar

    y_hat = b0 + b1 * x
    resid = y - y_hat

    SSE = np.sum(resid**2)
    SST = np.sum((y - y_bar)**2)
    R2  = 1 - SSE/SST

    # Phương sai phần dư & sai số chuẩn hệ số
    df = n - 2
    s2 = SSE / df
    se_b1 = (s2 / Sxx) ** 0.5
    se_b0 = (s2 * (1/n + x_bar**2 / Sxx)) ** 0.5

    return {
        "b0": b0, "b1": b1, "y_hat": y_hat, "resid": resid,
        "SSE": SSE, "SST": SST, "R2": R2,
        "x_bar": x_bar, "y_bar": y_bar, "Sxx": Sxx, "Sxy": Sxy,
        "se_b0": se_b0, "se_b1": se_b1, "s2": s2, "df": df, "n": n
    }

def print_summary(title, x, y, res):
    st.write("="*80)
    st.write(title)
    st.write("- Dữ liệu x:", list(x))
    st.write("- Dữ liệu y:", list(y))
    st.write(f"- n = {res['n']}, x̄ = {res['x_bar']:.4f}, ȳ = {res['y_bar']:.4f}")
    st.write(f"- Sxx = {res['Sxx']:.4f}, Sxy = {res['Sxy']:.4f}")
    st.write(f"- Hệ số: b0 = {res['b0']:.6f}, b1 = {res['b1']:.6f}")
    st.write(f"- Mô hình: ŷ = {res['b0']:.6f} + {res['b1']:.6f} * x")
    st.write(f"- SSE = {res['SSE']:.6f}, SST = {res['SST']:.6f}, R² = {res['R2']:.6f}")
    st.write(f"- se(b0) = {res['se_b0']:.6f}, se(b1) = {res['se_b1']:.6f}, df = {res['df']}")
    # Bảng tóm tắt
    st.write("\n(x, y) -> (ŷ, e):")
    for xi, yi, yhat, ei in zip(x, y, res['y_hat'], res['resid']):
        st.write(f"({xi:>4}, {yi:>6}) -> (ŷ={yhat:>8.4f}, e={ei:>+9.4f})")
    

def save_scatter_with_line(outpath, title, x, y, b0, b1):
    # Lưu ý: theo yêu cầu, không set màu cụ thể và mỗi chart là một figure riêng
    plt.figure()
    plt.scatter(x, y, label="Dữ liệu")
    xgrid = np.linspace(min(x), max(x), 100)
    yfit = b0 + b1 * xgrid
    plt.plot(xgrid, yfit, label="Đường hồi quy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    st.pyplot(plt)
    plt.close()

# =============================
# BÀI TẬP 1 – Điểm thi & thời gian học
# =============================
x1 = [1, 2, 3, 4, 5, 6, 7, 8]               # số giờ học
y1 = [52, 55, 60, 63, 67, 72, 74, 78]       # điểm thi

res1 = simple_linreg(x1, y1)
print_summary("Bài tập 1 – Điểm thi & thời gian học", x1, y1, res1)

# Ví dụ dự báo tại x = 6.5
x_new_1 = 6.5
y_pred_1 = res1["b0"] + res1["b1"] * x_new_1
print(f"Dự báo điểm thi tại x = {x_new_1}: ŷ = {y_pred_1:.4f}\n")

# Vẽ hình và lưu
save_scatter_with_line(
    outpath="/mnt/data/ex1_hoi_quy_don_bien.png",
    title="Bài 1: Điểm thi & thời gian học (hồi quy tuyến tính)",
    x=x1, y=y1, b0=res1["b0"], b1=res1["b1"]
)

# =============================
# BÀI TẬP 2 – Quảng cáo & doanh số
# =============================
x2 = [2, 3, 4, 5, 6, 8, 10, 12]             # quảng cáo (nghìn USD)
y2 = [14, 16, 21, 23, 26, 30, 34, 38]       # doanh số (nghìn sp)

res2 = simple_linreg(x2, y2)
print_summary("Bài tập 2 – Quảng cáo & doanh số", x2, y2, res2)

# Kiểm định ý nghĩa b1 (t = b1 / se(b1))
t_b1 = res2["b1"] / res2["se_b1"]
print(f"Kiểm định H0: b1 = 0 -> t = {t_b1:.4f} với df = {res2['df']} (mức ý nghĩa 5% tham chiếu t_{{0.975, df}})")

# Dự báo tại x = 9 (nghìn USD)
x_new_2 = 9
y_pred_2 = res2["b0"] + res2["b1"] * x_new_2
print(f"Dự báo doanh số tại x = {x_new_2}: ŷ = {y_pred_2:.4f} (nghìn sp)\n")

# Vẽ hình và lưu
save_scatter_with_line(
    outpath="/mnt/data/ex2_hoi_quy_don_bien.png",
    title="Bài 2: Quảng cáo & doanh số (hồi quy tuyến tính)",
    x=x2, y=y2, b0=res2["b0"], b1=res2["b1"]
)

# =============================
# LƯU BẢNG KẾT QUẢ RA CSV (tùy chọn)
# =============================
df1 = pd.DataFrame({"x": x1, "y": y1, "y_hat": res1["y_hat"], "resid": res1["resid"]})
df2 = pd.DataFrame({"x": x2, "y": y2, "y_hat": res2["y_hat"], "resid": res2["resid"]})
df1.to_csv("data/ex1_ket_qua.csv", index=False)
df2.to_csv("data/ex2_ket_qua.csv", index=False)

# print("Đã lưu hình và CSV vào /mnt/data/:")
# print(" - ex1_hoi_quy_don_bien.png, ex1_ket_qua.csv")
# print(" - ex2_hoi_quy_don_bien.png, ex2_ket_qua.csv")