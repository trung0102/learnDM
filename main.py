import streamlit as st
import pandas as pd

st.title("Ứng dụng DataMining đầu tiên")
st.write("Xin chào, đây là Web App đầu tiên dùng Streamlit!")

# Đọc dữ liệu từ csv
df = pd.read_csv("datasets/data_extended.csv")

st.write("Dữ liệu gốc:")

st.dataframe(df)

st.write("Trung bình từng học sinh:")

df["Trung bình"] = df[["Điểm Toán","Điểm Văn","Điểm Anh"]].mean(axis=1)

st.dataframe(df)

st.write("Trung bình theo môn:")
avg_subjects = df[["Điểm Toán","Điểm Văn","Điểm Anh"]].mean()

st.dataframe(avg_subjects)

st.bar_chart(avg_subjects)

st.line_chart(df["Trung bình"])
