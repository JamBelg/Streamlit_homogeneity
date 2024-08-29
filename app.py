import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import f_oneway, levene, bartlett
import plotly.express as px

# Function to detect outliers within each group using IsolationForest
def detect_outliers_isolation_forest(data, value_col, group_col):
    """
    Test the presence of outliers by isolationForest

    :param data: dataframe
    :param value_col: value_col (string) is the name if tested variable (column if dataframe)
    :param group_col: group_col (categorical column) will be used to group data
    :return: returns detected outliers
    """ 
    data['is_outlier'] = False
    for level in data[group_col].unique():
        group_data = data[data[group_col] == level].copy()
        if len(group_data) > 1:
            model = IsolationForest(contamination=0.01, random_state=42)
            group_data['is_outlier'] = model.fit_predict(group_data[[value_col]]) == -1
            data.loc[group_data.index, 'is_outlier'] = group_data['is_outlier']
    outliers = data[data['is_outlier']==True].iloc[:,:-1]
    return outliers


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
    
    st.markdown("---")

    if tested!='' and len(selected_vars)>0:
        
        # Outliers
        st.markdown("<b>Outliers</b>", unsafe_allow_html=True)
        
        ## IQR method
        st.markdown("<u>IQR-method</u>:", unsafe_allow_html=True)
        outliers_iqr = pd.DataFrame()
        for var in selected_vars:
            for level in data[var].unique():
                group_data = data[data[var] == level].copy()
                Q1 = group_data[tested].quantile(0.25)
                Q3 = group_data[tested].quantile(0.75)
                IQR = Q3-Q1
                outliers_i = group_data[(group_data[tested]>Q3+1.2*IQR) | (group_data[tested]<Q1-1.2*IQR)]
                if outliers_iqr.shape[1]>0:
                    outliers_iqr = pd.concat([outliers_iqr, outliers_i]).drop_duplicates()
                else:
                    outliers_iqr = outliers_i
        if outliers_iqr.shape[1]>0:
            outliers_iqr = outliers_iqr[selected_vars.tolist().append(tested)]
            st.markdown(outliers_iqr.style.hide(axis="index").to_html(), 
                        unsafe_allow_html=True)
        else:
            st.markdown("No outliers detected !", unsafe_allow_html=True)
        
        ## Z-score
        # Define a threshold for identifying outliers (Z-score threshold 2 - 3)
        threshold = 2
        st.markdown("<u>Z-score</u> (threshold "+str(threshold)+"):", unsafe_allow_html=True)
        # Calculate the Z-scores
        z_scores = (data[tested] - data[tested].mean()) / data[tested].std()
 
        # Identify the outliers
        outliers_zscore = data[abs(z_scores) > threshold]
 
        # Print the outliers
        if outliers_zscore.shape[0]>0:
            st.markdown(outliers_zscore.style.hide(axis="index").to_html(), 
                        unsafe_allow_html=True)
            #st.table(outliers_zscore)
            #st.write(outliers[tested].tolist())
        else:
            st.markdown("No outliers detected !", unsafe_allow_html=True)
        
        ## IsolationForest
        st.markdown('<u>Isolation Forest</u>:',unsafe_allow_html=True)
        i=0
        for var in selected_vars:
            i+=1
            outliers_i = detect_outliers_isolation_forest(data, tested, var)
            if i==1:
                outliers_isolForest = outliers_i
            else:
                outliers_isolForest = pd.concat([outliers_isolForest, outliers_i]).drop_duplicates()
        if(outliers_isolForest.shape[1]>0):
            st.markdown(outliers_isolForest.style.hide(axis="index").to_html(), 
                        unsafe_allow_html=True)
            #st.table(outliers_isolForest)
        else:
            st.write('No outliers')
        
        if st.checkbox('Delete outliers'):
            outliers = pd.concat([outliers_zscore, outliers_isolForest, outliers_iqr]).drop_duplicates()
            data=data[~data.index.isin(outliers.index)]
        
        st.markdown("---")
        
        
        # Plots
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
            if anova_result.pvalue>=0.05:
                st.markdown("p-value ≥ 0.05: Fail to reject the null hypothesis. The groups are homogeneous.", unsafe_allow_html=True)
            else:
                st.markdown("p-value < 0.05: Reject the null hypothesis. The groups are not homogeneous.", unsafe_allow_html=True)
            
        
        st.markdown("<li>Levene test</li>", unsafe_allow_html=True)
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform Levene test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            levene_result = levene(*groups)
            st.markdown("by "+var+":", unsafe_allow_html=True)
            st.write(levene_result)
            if levene_result.pvalue>=0.05:
                st.markdown("p-value ≥ 0.05: Fail to reject H₀, the groups are homogeneous in terms of variance.", unsafe_allow_html=True)
            else:
                st.markdown("p-value < 0.05: Reject H₀, the groups are not homogeneous in terms of variance.", unsafe_allow_html=True)

        st.markdown("<li>Bartlett test</li>", unsafe_allow_html=True)
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform Bartlett test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            bartlett_result = bartlett(*groups)
            st.markdown("by "+var+":", unsafe_allow_html=True)
            st.write(bartlett_result)
            if bartlett_result.pvalue>=0.05:
                st.markdown("p-value ≥ 0.05: Fail to reject H₀, the groups are homogeneous in terms of variance.", unsafe_allow_html=True)
            else:
                st.markdown("p-value < 0.05: Reject H₀, the groups are not homogeneous in terms of variance.", unsafe_allow_html=True)
        
        st.markdown("</ul>", unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()
