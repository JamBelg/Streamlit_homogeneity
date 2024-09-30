# Streamlit_homogeneity
## A Streamlit-based application for testing homogeneity across multiple variables using statistical methods.

[![Link of the application](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://homogeneity.streamlit.app/)

This application offers a comprehensive solution for analyzing data homogeneity across multiple variables through ANOVA, Levene, and Bartlett tests. Users can upload a CSV file or use the provided sample dataset to examine data uniformity and identify outliers using advanced techniques such as Z-score, IQR (Interquartile Range), and Isolation Forest.

**Key Features**

 - Outlier Detection: Utilize Z-score, IQR, and Isolation Forest methods to detect and handle outliers in your data.
 - Interactive Visualizations: Gain insights into data distribution and variance with interactive plots, including boxplots, Z-score plots, and Q-Q plots.
 - Data Analysis and Cleaning: Easily remove detected outliers to refine your dataset for more accurate analysis.
 - Homogeneity Testing: Perform ANOVA, Levene, and Bartlett tests to assess group uniformity and variance, with detailed results and summaries provided for interpretation.

**Screenshots**

 - Outlier Detection:

<p align="center"> <img src="Screenshot 2024-09-03 at 06.52.16.png" width="300"> </p>

 - Data Visualization:

<p align="center"> <img src="Screenshot 2024-09-03 at 06.52.37.png" width="500"> </p>

 - Homogeneity Testing:

<p align="center"> <img src="Screenshot 2024-09-03 at 06.52.46.png" width="500"> </p>

**Getting Started**
Install the necessary libraries:

 - streamlit
 - pandas
 - numpy
 - scipy
 - sklearn
 - plotly

**Run the application:**
```
streamlit run app.py
```

Upload your data or use the sample data provided to begin analysis.

**Conclusion**

This tool is ideal for data scientists, analysts, and researchers who need to assess the homogeneity of their datasets, detect outliers, and visualize data distributions interactively. Start exploring your data's homogeneity today with the Streamlit Homogeneity Testing Tool!
