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
