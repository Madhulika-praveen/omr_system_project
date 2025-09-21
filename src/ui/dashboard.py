import streamlit as st
from omr.omr_processor import process_sheet

st.title("OMR Evaluation Dashboard")

uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpeg", "jpg", "png"])
set_id = st.text_input("Set ID (e.g., A)", value="A")

if uploaded_file and st.button("Evaluate"):
    result = process_sheet(uploaded_file, set_id.upper())
    st.write(result)
