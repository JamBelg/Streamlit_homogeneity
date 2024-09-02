import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from scipy.stats import f_oneway, levene, bartlett
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    
        float_columns = data.select_dtypes(include=['float']).columns  # Select only float columns
        columns_to_round = [col for col in float_columns if col not in ['X', 'Y', 'Z']]  # Exclude 'X', 'Y', 'Z'

        # Round the selected columns to two decimal places
        data[columns_to_round] = data[columns_to_round].round(2)
    
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
        
        list_columns = selected_vars.copy()
        list_columns.append(tested)
        
        # Outliers
        #st.markdown("<b>Outliers</b>", unsafe_allow_html=True)
        
        ## IQR method
        #st.markdown("<u>IQR-method</u>:", unsafe_allow_html=True)
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

        
        ## Z-score
        # Define a threshold for identifying outliers (Z-score threshold 2 - 3)
        threshold = 2
        #st.markdown("<u>Z-score</u> (threshold "+str(threshold)+"):", unsafe_allow_html=True)
        # Calculate the Z-scores
        z_scores = (data[tested] - data[tested].mean()) / data[tested].std()
 
        # Identify the outliers
        outliers_zscore = data[abs(z_scores) > threshold]
 
        
        ## IsolationForest
        #st.markdown('<u>Isolation Forest</u>:',unsafe_allow_html=True)
        i=0
        for var in selected_vars:
            i+=1
            outliers_i = detect_outliers_isolation_forest(data, tested, var)
            if i==1:
                outliers_isolForest = outliers_i
            else:
                outliers_isolForest = pd.concat([outliers_isolForest, outliers_i]).drop_duplicates()
        
        
        # Add a column to each DataFrame to indicate that the outlier was detected by the respective method
        outliers_zscore['Z-score'] = 'x'
        outliers_isolForest['IsolationForest'] = 'x'
        outliers_iqr['IQR method'] = 'x'


        # Merge the dataframes on categorical features
        list_columns = selected_vars.copy()
        list_columns.append(tested)
        merged_df = pd.merge(outliers_zscore[list_columns+['Z-score']], 
                            outliers_isolForest[list_columns+['IsolationForest']], 
                            on=list_columns, 
                            how='outer')

        merged_df = pd.merge(merged_df, 
                            outliers_iqr[list_columns+['IQR method']], 
                            on=list_columns, 
                            how='outer')

        # Fill NaN values with an empty string (because not all methods will detect an outlier for each row)
        merged_df.fillna('', inplace=True)
        merged_df = merged_df.drop_duplicates(subset=list_columns)
        merged_df[tested] = merged_df[tested].round(2)
        
        if merged_df.shape[1]>0:
            # Format the 'tested' column to two decimal places using the Styler object
            styled_df = merged_df.style.format({tested: "{:.2f}"}).hide(axis="index")

            # Render the styled DataFrame in Streamlit
            st.markdown(styled_df.to_html(), unsafe_allow_html=True)
            #st.markdown(merged_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

            if st.checkbox('Delete outliers'):
                outliers = pd.concat([outliers_zscore, outliers_isolForest, outliers_iqr]).drop_duplicates()
                data=data[~data.index.isin(outliers.index)]
        else:
            st.markdown('No outliers !!', unsafe_allow_html=True)
            
        st.markdown("---")
        
        
        # Plots
        st.markdown("<b>Vizualisation</b>", unsafe_allow_html=True)
        modified_list = [f'Boxplot of {item}' for item in selected_vars]
        modified_list = modified_list+['Z-score', 'Density']
        tabs = st.tabs(modified_list)
        for i, tab in enumerate(tabs):
            with tab:
                # Boxplots
                if i<len(modified_list)-2:
                    fig = px.box(data,
                                x=selected_vars[i],
                                y=tested,
                                color=selected_vars[i],
                                title='Boxplot '+tested+' vs '+selected_vars[i])

                    # Update layout (optional)
                    fig.update_layout(
                        yaxis_title=tested,
                        xaxis_title=selected_vars[i],
                        showlegend=False
                    )

                    # Display the boxplot in Streamlit
                    st.plotly_chart(fig,
                                    theme = "streamlit",
                                    use_container_width = True)
                # Z-score
                elif i==len(modified_list)-2:
                    # Create a Plotly figure
                    fig = go.Figure()

                    # Add Z-score scatter plot
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=z_scores,
                        text = [f"{tested}: {value:.2f}<br>Z-score: {zs:.2f}" for value, zs in zip(data[tested], z_scores)],
                        mode='markers',
                        name='Z-Score'
                    ))

                    # Add threshold lines at +/- 2
                    fig.add_trace(go.Scatter(
                        x=[0, len(z_scores) - 1],
                        y=[2, 2],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='+2 Threshold'
                    ))

                    fig.add_trace(go.Scatter(
                        x=[0, len(z_scores) - 1],
                        y=[-2, -2],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='-2 Threshold'
                    ))

                    # Customize the layout
                    fig.update_layout(
                        title="Z-Score Plot with +/- 2 Thresholds",
                        xaxis_title="Index",
                        yaxis_title="Z-Score",
                        showlegend=True
                    )

                    # Display the Plotly figure in Streamlit
                    st.plotly_chart(fig)
                elif i==len(modified_list)-1:
                    # Create a histogram and density plot with Plotly
                    fig = px.histogram(data, x=tested, marginal='box', title=f'Distribution of {tested}')
                    density = stats.gaussian_kde(data[tested].dropna())
                    x_vals = np.linspace(data[tested].min(), data[tested].max(), 20)
                    fig.add_scatter(x=x_vals, y=density(x_vals), mode='lines', name='Density')
                    st.plotly_chart(fig, use_container_width=True)

                    # Q-Q Plot with Matplotlib
                    st.subheader(f'Q-Q Plot of {tested}')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    stats.probplot(data[tested].dropna(), dist="norm", plot=ax)
                    ax.set_title(f'Q-Q Plot of {tested}')
                    st.pyplot(fig)
        
        st.markdown("---")
        
        
        st.markdown("<b>Homogeneity tests</b>", unsafe_allow_html=True)
        st.markdown("<ul>", unsafe_allow_html=True)

        st.markdown("<li>Anova test</li>", unsafe_allow_html=True)
        tab_anov1, tab_anov2 = st.tabs(["Resume", "Details"])
        anova_results=[]
        anova_resume=[]
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform ANOVA test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            anova_result_i = f_oneway(*groups)
            anova_results.append(anova_result_i)
            if anova_result_i.pvalue>=0.05:
                anova_resume.append('p-value ≥ 0.05: Fail to reject H₀: There is no significant difference in the means, suggesting uniformity between the groups')
            else:
                anova_resume.append('p-value < 0.05: Reject H₀: There is a significant difference in the means, indicating variation between the groups.')
        with tab_anov1:
            anova_table = pd.DataFrame({'Variable':selected_vars,
                                        'Results': anova_resume})
            #st.table(anova_table)
            st.markdown(anova_table.style.hide(axis="index").to_html(), 
                        unsafe_allow_html=True)

        with tab_anov2:
            st.markdown("<ol>", unsafe_allow_html=True)
            for i, var in enumerate(selected_vars):
                st.markdown('<li>Anova test by <u>'+var+"</u>:", unsafe_allow_html=True)
                st.write(anova_results[i])
                st.markdown("</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)
            
        
        # Levene test
        st.markdown("<li>Levene test</li>", unsafe_allow_html=True)
        tab_levene1, tab_levene2 = st.tabs(['Resume', 'Details'])
        levene_results=[]
        levene_resume=[]
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform Levene test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            levene_result_i = levene(*groups)
            levene_results.append(levene_result_i)
            if levene_result_i.pvalue>=0.05:
                levene_resume.append("p-value ≥ 0.05: Fail to reject H₀, the groups are homogeneous in terms of variance.")
            else:
                levene_resume.append("p-value < 0.05: Reject H₀, the groups are not homogeneous in terms of variance.")

        with tab_levene1:
            levene_table = pd.DataFrame({'Variable':selected_vars,
                                        'Results': levene_resume})
            st.markdown(levene_table.style.hide(axis="index").to_html(), 
                        unsafe_allow_html=True)
        with tab_levene2:
            st.markdown("<ol>", unsafe_allow_html=True)
            for i, var in enumerate(selected_vars):
                st.markdown('<li>Levene test by <u>'+var+"</u>:", unsafe_allow_html=True)
                st.write(levene_results[i])
                st.markdown("</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)


        # Bartlett test
        st.markdown("<li>Bartlett test</li>", unsafe_allow_html=True)
        tab_bartlett1, tab_bartlett2 = st.tabs(['Resume', 'Details'])
        bartlett_results=[]
        bartlett_resume=[]
        for var in selected_vars:
            data[var] = data[var].astype('category')
            # Perform Bartlett test
            groups = [data[tested][data[var] == level] for level in data[var].unique()]
            bartlett_result_i = bartlett(*groups)
            bartlett_results.append(bartlett_result_i)
            if bartlett_result_i.pvalue>=0.05:
                bartlett_resume.append("p-value ≥ 0.05: Fail to reject H₀, the groups are homogeneous in terms of variance.")
            else:
                bartlett_resume.append("p-value < 0.05: Reject H₀, the groups are not homogeneous in terms of variance.")
        
        with tab_bartlett1:
            bartlett_table = pd.DataFrame({'Variable':selected_vars,
                                           'Results': bartlett_resume})
            st.markdown(bartlett_table.style.hide(axis="index").to_html(), 
                        unsafe_allow_html=True)
        with tab_bartlett2:
            st.markdown("<ol>", unsafe_allow_html=True)
            for i, var in enumerate(selected_vars):
                st.markdown('<li>Bartlett test by <u>'+var+"</u>:", unsafe_allow_html=True)
                st.write(bartlett_results[i])
                st.markdown("</li>", unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)
        
        st.markdown("</ul>", unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()
