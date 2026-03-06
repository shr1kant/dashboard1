#!/usr/bin/env python3
"""
app.py - AI Diagnostic Tool Adoption Dashboard

Interactive Streamlit dashboard for analyzing physician AI diagnostic tool
adoption patterns in Singapore hospitals.

Features:
- Overview KPIs and metrics
- Correlation analysis with heatmap
- Demographic breakdowns with bar charts
- Scatter plot analysis
- OLS regression prediction model

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Diagnostic Adoption Dashboard - SG Hospitals",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A5F;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(filepath: str = 'adoption_data.csv') -> pd.DataFrame:
    """
    Load and cache the adoption data.
    """
    try:
        df = pd.read_csv(filepath)
        df['specialty'] = df['specialty'].astype('category')
        df['hospital_size'] = df['hospital_size'].astype('category')
        df['age_group'] = df['age_group'].astype('category')
        df['experience_level'] = df['experience_level'].astype('category')
        return df
    except FileNotFoundError:
        st.error(f"Data file '{filepath}' not found. Please run 'generate_data.py' first.")
        st.stop()


def filter_data(
    df: pd.DataFrame,
    age_range: tuple,
    specialties: list,
    hospital_sizes: list
) -> pd.DataFrame:
    """
    Apply filters to the dataset.
    """
    filtered = df.copy()
    filtered = filtered[
        (filtered['age'] >= age_range[0]) & 
        (filtered['age'] <= age_range[1])
    ]
    if specialties:
        filtered = filtered[filtered['specialty'].isin(specialties)]
    if hospital_sizes:
        filtered = filtered[filtered['hospital_size'].isin(hospital_sizes)]
    return filtered

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap for TAM scores."""
    score_cols = ['age', 'years_experience', 'pu_score', 'eou_score', 
                  'trust_score', 'ita_score', 'overall_tam_score']
    corr_matrix = df[score_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Correlation Matrix - TAM Constructs & Demographics', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
        height=500,
        xaxis_title='',
        yaxis_title='',
        xaxis={'tickangle': 45}
    )
    return fig


def create_score_by_demographic_chart(df: pd.DataFrame, demographic: str) -> go.Figure:
    """Create bar chart showing scores by demographic category."""
    score_cols = ['pu_score', 'eou_score', 'trust_score', 'ita_score']
    grouped = df.groupby(demographic)[score_cols].mean().reset_index()
    melted = grouped.melt(id_vars=[demographic], value_vars=score_cols, var_name='Score Type', value_name='Mean Score')
    
    melted['Score Type'] = melted['Score Type'].map({
        'pu_score': 'Perceived Usefulness',
        'eou_score': 'Ease of Use',
        'trust_score': 'Trust',
        'ita_score': 'Intention to Adopt'
    })
    
    fig = px.bar(
        melted, x=demographic, y='Mean Score', color='Score Type',
        barmode='group', color_discrete_sequence=px.colors.qualitative.Set2,
        title=f'Average TAM Scores by {demographic.replace("_", " ").title()}'
    )
    
    fig.update_layout(height=450, xaxis_title=demographic.replace('_', ' ').title(),
                      yaxis_title='Mean Score (1-5)', yaxis_range=[0, 5], legend_title='Score Type', hovermode='x unified')
    fig.add_hline(y=3.5, line_dash="dash", line_color="red",
                  annotation_text="Adoption Threshold (3.5)", annotation_position="top right")
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> go.Figure:
    """Create interactive scatter plot with trend line."""
    col_names = {
        'pu_score': 'Perceived Usefulness', 'eou_score': 'Ease of Use',
        'trust_score': 'Trust', 'ita_score': 'Intention to Adopt'
    }
    
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, trendline='ols', opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set1,
        title=f'{col_names.get(y_col, y_col)} vs {col_names.get(x_col, x_col)}',
        hover_data=['physician_id', 'age', 'specialty']
    )
    
    fig.update_layout(height=500, xaxis_title=col_names.get(x_col, x_col),
                      yaxis_title=col_names.get(y_col, y_col), xaxis_range=[0.5, 5.5], yaxis_range=[0.5, 5.5])
    fig.add_hline(y=3.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=3.5, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def create_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Create histogram with KDE for score distribution."""
    col_names = {
        'pu_score': 'Perceived Usefulness', 'eou_score': 'Ease of Use',
        'trust_score': 'Trust', 'ita_score': 'Intention to Adopt'
    }
    
    fig = px.histogram(df, x=column, nbins=30, marginal='box',
                       color_discrete_sequence=['#667eea'],
                       title=f'Distribution of {col_names.get(column, column)}')
    fig.update_layout(height=400, xaxis_title=col_names.get(column, column), yaxis_title='Count', showlegend=False)
    
    mean_val = df[column].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
    return fig

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

@st.cache_data
def run_ols_regression(df: pd.DataFrame) -> dict:
    """Run OLS regression: ITA ~ PU + EOU + Trust"""
    X = df[['pu_score', 'eou_score', 'trust_score']]
    y = df['ita_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'coefficients': {'Perceived Usefulness': model.coef_[0], 'Ease of Use': model.coef_[1], 'Trust': model.coef_[2]},
        'intercept': model.intercept_,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'X_test': X_test, 'y_test': y_test, 'y_pred_test': y_pred_test
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    
    st.markdown('<h1 class="main-header">🏥 AI Diagnostic Adoption Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyzing Physician Technology Acceptance in Singapore Hospitals</p>', unsafe_allow_html=True)
    
    df = load_data()
    
    # SIDEBAR FILTERS
    with st.sidebar:
        st.title("🔍 Filters")
        st.markdown("---")
        
        st.subheader("Age Range")
        age_range = st.slider("Select age range:", min_value=int(df['age'].min()),
                              max_value=int(df['age'].max()), value=(25, 65), key='age_slider')
        
        st.markdown("---")
        st.subheader("Specialty")
        all_specialties = df['specialty'].unique().tolist()
        selected_specialties = st.multiselect("Select specialties:", options=all_specialties,
                                               default=all_specialties, key='specialty_select')
        
        st.markdown("---")
        st.subheader("Hospital Size")
        all_hospital_sizes = df['hospital_size'].unique().tolist()
        selected_hospital_sizes = st.multiselect("Select hospital sizes:", options=all_hospital_sizes,
                                                  default=all_hospital_sizes, key='hospital_select')
        
        st.markdown("---")
        filtered_df = filter_data(df, age_range, selected_specialties, selected_hospital_sizes)
        st.metric("Filtered Sample Size", f"{len(filtered_df):,}")
        st.caption(f"of {len(df):,} total records")
        
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()
    
    # MAIN CONTENT - TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔗 Correlations", "📈 Demographics", "🎯 Scatter Analysis", "🤖 Prediction Model"])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown('<h2 class="section-header">Key Performance Indicators</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_pu = filtered_df['pu_score'].mean()
            st.metric(label="Avg Perceived Usefulness", value=f"{avg_pu:.2f}",
                      delta=f"{avg_pu - 3.5:.2f} vs threshold", delta_color="normal")
        with col2:
            avg_eou = filtered_df['eou_score'].mean()
            st.metric(label="Avg Ease of Use", value=f"{avg_eou:.2f}",
                      delta=f"{avg_eou - 3.5:.2f} vs threshold", delta_color="normal")
        with col3:
            avg_trust = filtered_df['trust_score'].mean()
            st.metric(label="Avg Trust Score", value=f"{avg_trust:.2f}",
                      delta=f"{avg_trust - 3.5:.2f} vs threshold", delta_color="normal")
        with col4:
            avg_ita = filtered_df['ita_score'].mean()
            st.metric(label="Avg Intention to Adopt", value=f"{avg_ita:.2f}",
                      delta=f"{avg_ita - 3.5:.2f} vs threshold", delta_color="normal")
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            adoption_rate = (filtered_df['ita_score'] > 3.5).mean() * 100
            st.metric(label="High Adoption Intent Rate", value=f"{adoption_rate:.1f}%", help="Percentage with ITA score > 3.5")
        with col2:
            st.metric(label="Sample Size", value=f"{len(filtered_df):,}", help="Number of physicians in filtered data")
        with col3:
            st.metric(label="Average Age", value=f"{filtered_df['age'].mean():.1f} years")
        with col4:
            st.metric(label="Avg Experience", value=f"{filtered_df['years_experience'].mean():.1f} years")
        
        st.markdown("---")
        st.markdown('<h2 class="section-header">Score Distributions</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_distribution_plot(filtered_df, 'pu_score'), use_container_width=True)
            st.plotly_chart(create_distribution_plot(filtered_df, 'trust_score'), use_container_width=True)
        with col2:
            st.plotly_chart(create_distribution_plot(filtered_df, 'eou_score'), use_container_width=True)
            st.plotly_chart(create_distribution_plot(filtered_df, 'ita_score'), use_container_width=True)
        
        st.markdown('<h2 class="section-header">Summary Statistics</h2>', unsafe_allow_html=True)
        summary_cols = ['age', 'years_experience', 'pu_score', 'eou_score', 'trust_score', 'ita_score', 'overall_tam_score']
        st.dataframe(filtered_df[summary_cols].describe().T.round(2), use_container_width=True, height=300)
    
    # TAB 2: CORRELATIONS
    with tab2:
        st.markdown('<h2 class="section-header">Correlation Analysis</h2>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box"><strong>📌 Key Findings:</strong><br>
        • EOU shows negative correlation with age (older physicians find AI tools harder to use)<br>
        • ITA is positively correlated with all TAM constructs (PU, EOU, Trust)<br>
        • Trust and EOU show moderate positive correlation</div>""", unsafe_allow_html=True)
        
        st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)
        
        st.markdown('<h3 class="section-header">Detailed Correlations with ITA</h3>', unsafe_allow_html=True)
        ita_correlations = filtered_df[['pu_score', 'eou_score', 'trust_score', 'age', 'years_experience']].corrwith(filtered_df['ita_score']).sort_values(ascending=False)
        corr_df = pd.DataFrame({'Variable': ita_correlations.index, 'Correlation with ITA': ita_correlations.values})
        corr_df['Strength'] = corr_df['Correlation with ITA'].apply(lambda x: 'Strong' if abs(x) > 0.5 else ('Moderate' if abs(x) > 0.3 else 'Weak'))
        corr_df['Direction'] = corr_df['Correlation with ITA'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
    
    # TAB 3: DEMOGRAPHICS
    with tab3:
        st.markdown('<h2 class="section-header">Scores by Demographics</h2>', unsafe_allow_html=True)
        demo_option = st.selectbox("Select demographic variable:", options=['specialty', 'hospital_size', 'age_group', 'experience_level'],
                                   format_func=lambda x: x.replace('_', ' ').title())
        st.plotly_chart(create_score_by_demographic_chart(filtered_df, demo_option), use_container_width=True)
        
        st.markdown("---")
        st.markdown('<h3 class="section-header">Detailed Breakdown</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("By Specialty")
            specialty_stats = filtered_df.groupby('specialty').agg({'ita_score': ['mean', 'std', 'count'], 'high_adoption_intent': 'mean'}).round(2)
            specialty_stats.columns = ['Mean ITA', 'Std Dev', 'Count', 'Adoption Rate']
            specialty_stats['Adoption Rate'] = (specialty_stats['Adoption Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(specialty_stats, use_container_width=True)
        with col2:
            st.subheader("By Hospital Size")
            hospital_stats = filtered_df.groupby('hospital_size').agg({'ita_score': ['mean', 'std', 'count'], 'high_adoption_intent': 'mean'}).round(2)
            hospital_stats.columns = ['Mean ITA', 'Std Dev', 'Count', 'Adoption Rate']
            hospital_stats['Adoption Rate'] = (hospital_stats['Adoption Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(hospital_stats, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Age Group Analysis")
        fig_age = px.box(filtered_df, x='age_group', y='ita_score', color='specialty',
                         title='Intention to Adopt by Age Group and Specialty', color_discrete_sequence=px.colors.qualitative.Set2)
        fig_age.update_layout(height=500)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # TAB 4: SCATTER ANALYSIS
    with tab4:
        st.markdown('<h2 class="section-header">Relationship Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        score_options = ['pu_score', 'eou_score', 'trust_score', 'ita_score']
        
        with col1:
            x_var = st.selectbox("X-axis variable:", options=score_options, index=0,
                                 format_func=lambda x: {'pu_score': 'Perceived Usefulness', 'eou_score': 'Ease of Use', 'trust_score': 'Trust', 'ita_score': 'Intention to Adopt'}[x])
        with col2:
            y_var = st.selectbox("Y-axis variable:", options=score_options, index=3,
                                 format_func=lambda x: {'pu_score': 'Perceived Usefulness', 'eou_score': 'Ease of Use', 'trust_score': 'Trust', 'ita_score': 'Intention to Adopt'}[x])
        with col3:
            color_var = st.selectbox("Color by:", options=['specialty', 'hospital_size', 'age_group'],
                                     format_func=lambda x: x.replace('_', ' ').title())
        
        st.plotly_chart(create_scatter_plot(filtered_df, x_var, y_var, color_var), use_container_width=True)
        st.info(f"📊 Correlation coefficient: **{filtered_df[x_var].corr(filtered_df[y_var]):.3f}**")
        
        st.markdown("---")
        st.markdown('<h3 class="section-header">Perceived Usefulness vs Intention to Adopt</h3>', unsafe_allow_html=True)
        fig_pu_ita = px.scatter(filtered_df, x='pu_score', y='ita_score', color='specialty', size='trust_score',
                                trendline='ols', opacity=0.7, title='PU vs ITA (bubble size = Trust score)',
                                color_discrete_sequence=px.colors.qualitative.Set1, hover_data=['physician_id', 'age', 'hospital_size'])
        fig_pu_ita.update_layout(height=500)
        st.plotly_chart(fig_pu_ita, use_container_width=True)
    
    # TAB 5: PREDICTION MODEL
    with tab5:
        st.markdown('<h2 class="section-header">OLS Regression Model</h2>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box"><strong>📌 Model Specification:</strong><br>
        <code>ITA = β₀ + β₁(PU) + β₂(EOU) + β₃(Trust) + ε</code><br><br>
        This model predicts Intention to Adopt (ITA) based on the three TAM constructs.</div>""", unsafe_allow_html=True)
        
        results = run_ols_regression(filtered_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² (Train)", f"{results['r2_train']:.3f}")
        with col2:
            st.metric("R² (Test)", f"{results['r2_test']:.3f}")
        with col3:
            st.metric("RMSE (Train)", f"{results['rmse_train']:.3f}")
        with col4:
            st.metric("RMSE (Test)", f"{results['rmse_test']:.3f}")
        
        st.markdown("---")
        st.markdown('<h3 class="section-header">Model Coefficients</h3>', unsafe_allow_html=True)
        
        coef_df = pd.DataFrame({'Variable': ['Intercept'] + list(results['coefficients'].keys()),
                                'Coefficient': [results['intercept']] + list(results['coefficients'].values())})
        coef_df['Coefficient'] = coef_df['Coefficient'].round(4)
        
        fig_coef = px.bar(coef_df[coef_df['Variable'] != 'Intercept'], x='Variable', y='Coefficient',
                          color='Coefficient', color_continuous_scale='RdBu_r', title='Regression Coefficients (Impact on ITA)')
        fig_coef.update_layout(height=400)
        st.plotly_chart(fig_coef, use_container_width=True)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown('<h3 class="section-header">Actual vs Predicted Values</h3>', unsafe_allow_html=True)
        
        pred_df = pd.DataFrame({'Actual': results['y_test'], 'Predicted': results['y_pred_test']})
        fig_pred = px.scatter(pred_df, x='Actual', y='Predicted', opacity=0.6, title='Actual vs Predicted ITA Scores (Test Set)', trendline='ols')
        fig_pred.add_trace(go.Scatter(x=[1, 5], y=[1, 5], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        fig_pred.update_layout(height=500, xaxis_title='Actual ITA Score', yaxis_title='Predicted ITA Score', xaxis_range=[1, 5], yaxis_range=[1, 5])
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown("---")
        st.markdown('<h3 class="section-header">🎯 Interactive Prediction</h3>', unsafe_allow_html=True)
        st.write("Enter values to predict Intention to Adopt:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            input_pu = st.slider("Perceived Usefulness", 1.0, 5.0, 3.5, 0.1)
        with col2:
            input_eou = st.slider("Ease of Use", 1.0, 5.0, 3.2, 0.1)
        with col3:
            input_trust = st.slider("Trust", 1.0, 5.0, 3.0, 0.1)
        
        prediction = np.clip(results['model'].predict([[input_pu, input_eou, input_trust]])[0], 1, 5)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Predicted ITA Score", value=f"{prediction:.2f}",
                      delta="High Intent" if prediction > 3.5 else "Low Intent",
                      delta_color="normal" if prediction > 3.5 else "inverse")
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prediction, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Adoption Intent"},
                gauge={'axis': {'range': [1, 5]}, 'bar': {'color': "#667eea"},
                       'steps': [{'range': [1, 2.5], 'color': "#ffcccb"}, {'range': [2.5, 3.5], 'color': "#ffffcc"}, {'range': [3.5, 5], 'color': "#90EE90"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 3.5}}
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    # FOOTER
    st.markdown("---")
    st.markdown("""<div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 AI Diagnostic Tool Adoption Dashboard | Singapore Hospitals Study</p>
        <p style="font-size: 0.8rem;">Built with Streamlit • Data is synthetic for demonstration purposes</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()