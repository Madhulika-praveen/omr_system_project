import streamlit as st
import pandas as pd
import sys
import os

# --- Fix imports for Streamlit Cloud ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.main import process_sheet

st.set_page_config(page_title="OMR Evaluation Dashboard", layout="wide")
st.title("OMR Evaluation Dashboard")

# Upload OMR image
uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpeg", "jpg", "png"])

if uploaded_file:
    # Input for Set ID
    set_id = st.text_input("Set ID (e.g., A)", value="A")
    
    if st.button("Evaluate"):
        try:
            # Call process_sheet directly
            result = process_sheet(uploaded_file, set_id.upper())
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display total score
                st.success(f"Total Score: {result['total']}")
                
                # Convert scores dict to DataFrame for display and download
                scores_df = pd.DataFrame(list(result['scores'].items()), columns=["Subject", "Score"])
                st.dataframe(scores_df)
                
                # Provide CSV download
                csv_data = scores_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv_data, "results.csv", "text/csv")
        
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
