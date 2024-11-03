import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('BankChurners.csv')

# Set the title of the app
st.title('Bank Churners Analysis')

# Sidebar navigation with checkboxes
st.sidebar.title("Navigation")
data_overview = st.sidebar.checkbox("Data Overview")
eda = st.sidebar.checkbox("EDA")
visualization = st.sidebar.checkbox("Visualization")

if data_overview:
    st.header('Customer Data Overview')
    st.write(df.head())

if eda:
    st.header('Exploratory Data Analysis')
    
    # Demographics Visualization
    st.subheader('Churn Rate by Gender')
    gender_churn = sns.countplot(x='Gender', hue='Attrition_Flag', data=df)
    st.pyplot(gender_churn.figure)
    plt.clf()  # Clear the figure after displaying

    # Age Distribution
    st.subheader('Age Distribution')
    age_distribution = sns.histplot(df['Customer_Age'], bins=30)
    st.pyplot(age_distribution.figure)
    plt.clf()  # Clear the figure after displaying

    # User Input: Age Filter
    st.subheader('Filter Customers by Age')
    age_filter = st.slider('Select Age Range:', 18, 100, (18, 100))
    filtered_df = df[(df['Customer_Age'] >= age_filter[0]) & (df['Customer_Age'] <= age_filter[1])]
    st.write(filtered_df)

if visualization:
    st.header('Visualization')
    
    
    # Income Distribution by Attrition Status
    st.subheader('Customer Attrition by Income Category')
    income_summary = df.groupby(['Attrition_Flag', 'Income_Category']).size().unstack()
    plt.figure(figsize=(12, 6))
    income_summary.plot(kind='bar', stacked=True)
    plt.title('Customer Attrition by Income Category')
    plt.xlabel('Income Category')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.legend(title='Attrition Flag')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure after displaying

     # Create a summary table for income category and attrition
    income_summary = df[df['Attrition_Flag'] == 'Attrited Customer'].groupby('Income_Category').size()

    # Plotting the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(income_summary, labels=income_summary.index, autopct='%1.1f%%', startangle=90, 
            shadow=True, colors=sns.color_palette("pastel", len(income_summary)))
    plt.title('Income Category Distribution Among Attrited Customers')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add outline to the pie chart
    for wedge in plt.gca().patches:
        wedge.set_edgecolor('black')

    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure after displaying
    

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure after displaying
    

# Final notes
st.sidebar.title("About")
st.sidebar.info("This app analyzes customer churn based on demographic and financial factors.")
