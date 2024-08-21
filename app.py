import streamlit as st
import pandas as pd
from scipy.stats import f_oneway
import plotly.express as px

def main():
    uploaded_file = st.file_uploader(label = "Choose a CSV file",
                                     accept_multiple_files = False)
    
    if uploaded_file is None:
        data = pd.read_csv("data_example.csv", sep=';')
    else:
        data = pd.read_csv(uploaded_file)
    
    st.write(data.head())
    
    var = st.selectbox(label = 'Quantitative variable',
                       options = data.columns)
    
    fact = st.selectbox(label = 'Categorical factor',
                        options = data.columns)

    if var!=fact:
        fig = px.box(data,
                     x=fact,
                     y=var,
                     color=fact,
                     title='Boxplot')

        # Update layout (optional)
        fig.update_layout(
            yaxis_title='Categorical factor',
            xaxis_title='Variable',
            showlegend=False
        )

        # Display the boxplot in Streamlit
        st.plotly_chart(fig,
                        theme = "streamlit",
                        use_container_width = True)

if __name__ == "__main__":
    main()