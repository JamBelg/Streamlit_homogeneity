import streamlit as st
import pandas as pd
from scipy.stats import f_oneway, levene, bartlett
import plotly.express as px

def main():
    
    with st.sidebar:
        st.markdown("---")
        st.title("Homogeneity testing")
        st.markdown(
            """
            This application is using Anova test to check the homogeneity of data along multiple axes (x, y, z)
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        
        uploaded_file = st.file_uploader(label = "Choose a CSV file",
                                        accept_multiple_files = False)
    
        if uploaded_file is None:
            data = pd.read_csv("data_example.csv", sep=';')
        else:
            data = pd.read_csv(uploaded_file)
    
    
        var_list = ['']+data.columns.tolist()
        tested = st.selectbox(label = 'Quantitative variable to be tested',
                              options = var_list)
        
        # Multiselect for choosing one or more variables
        selected_vars = st.multiselect(
            'Select variables to plot:',
            options=data.columns.tolist()
        )
        
        
        
        st.markdown("---")
        st.markdown(
            """
            Connect with me:

            <a href="https://www.linkedin.com/in/jamel-belgacem-289606a7/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="30" height="30" alt="LinkedIn"/>
            </a>
            
            <a href="https://github.com/JamBelg" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="30" height="30" alt="GitHub"/>
            </a>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        

    if st.checkbox('Show raw data'):
        if uploaded_file is None:
            st.subheader('Raw data')
            
        else:
            st.subheader('Uploaded data')
        st.write(data.head(20))
    

    if tested!='' and len(selected_vars)>0:
        st.markdown("<b>Vizualisation</b>", unsafe_allow_html=True)
        for var in selected_vars:
            fig = px.box(data,
                        x=var,
                        y=tested,
                        color=var,
                        title='Boxplot '+tested+' vs '+var)

            # Update layout (optional)
            fig.update_layout(
                yaxis_title=tested,
                xaxis_title=var,
                showlegend=False
            )

            # Display the boxplot in Streamlit
            st.plotly_chart(fig,
                            theme = "streamlit",
                            use_container_width = True)
        st.markdown("---")
        st.markdown("<b>Outliers</b>", unsafe_allow_html=True)
        # Calculate the Z-scores
        z_scores = (data[tested] - data[tested].mean()) / data[tested].std()
 
        # Define a threshold for identifying outliers (e.g., Z-score threshold of 2)
        threshold = 3
 
        # Identify the outliers
        outliers = data[abs(z_scores) > threshold]
 
        # Print the outliers
        st.markdown("<u>Z-score</u> (threshold "+str(threshold)+"):", unsafe_allow_html=True)
        if outliers.shape[0]>0:
            st.write(outliers[tested].tolist())
        else:
            st.markdown("No outliers detected !", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<b>Homogeneity tests</b>", unsafe_allow_html=True)
        st.markdown("<ul>", unsafe_allow_html=True)

        st.markdown("<li>Anova test</li>", unsafe_allow_html=True)
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform ANOVA test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            anova_result = f_oneway(*groups)
            st.markdown("by "+var+":", unsafe_allow_html=True)
            st.write(anova_result)
            
        
        st.markdown("<li>Levene test</li>", unsafe_allow_html=True)
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform Levene test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            levene_result = levene(*groups)
            st.markdown("by "+var+":", unsafe_allow_html=True)
            st.write(levene_result)

        st.markdown("<li>Bartlett test</li>", unsafe_allow_html=True)
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform Bartlett test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            levene_result = bartlett(*groups)
            st.markdown("by "+var+":", unsafe_allow_html=True)
            st.write(levene_result)
        
        st.markdown("</ul>", unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()
