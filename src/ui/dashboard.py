import streamlit as st
import requests
import pandas as pd

st.title("OMR Evaluation Dashboard")

# Upload OMR image
uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpeg", "jpg", "png"])

if uploaded_file:
    # Input for Set ID
    set_id = st.text_input("Set ID (e.g., A)", value="A")
    
    if st.button("Evaluate"):
        # Prepare files and parameters for API request
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        params = {"set_id": set_id}
        
        # Send request to FastAPI OMR endpoint
        response = requests.post("http://127.0.0.1:8000/upload/", files=files, params=params)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display total score
            st.success(f"Total Score: {result['total']}")
            
            # Convert scores dict to DataFrame for display and download
            scores_df = pd.DataFrame(list(result['scores'].items()), columns=["Subject", "Score"])
            st.dataframe(scores_df)
            
            # Provide CSV download
            csv_data = scores_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv_data, "results.csv", "text/csv")
        
        else:
            st.error(f"Error: {response.text}")
