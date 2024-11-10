import streamlit as st
import pandas as pd
from clean_data import fetch_student_data, clean_data
from data_visualization import scatterplot
from preproc_data import prepare_data, feature_selection, normalization, pca
from regression import regression

def midtermPage():
    st.title("Midterm Checkpoint")
    
    data = fetch_student_data()
    cleaned_data = clean_data(data)

    X, y = prepare_data(cleaned_data)

    X_selected = feature_selection(X, y)
    X_normalized = normalization(X_selected)
    X_pca = pca(X_normalized)

    processed_data = pd.concat([pd.DataFrame(X_pca), cleaned_data[['G3']]], axis=1)

    series1, series2 = regression(processed_data)
    regression_data = pd.DataFrame({
        'Actual Grades': series1,
        'Predicted Grades': series2
    })
    st.line_chart(regression_data)

    st.header("Visualizations")
    scatterplot(regression_data)
    st.caption("Scatter Plot")
            
if __name__ == "__main__":
    midtermPage()