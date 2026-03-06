#!/usr/bin/env python3
"""
generate_data.py - Synthetic Data Generator for AI Diagnostic Tool Adoption Study

This script generates synthetic survey data simulating physician responses
regarding AI diagnostic tool adoption in Singapore hospitals.

Based on Technology Acceptance Model (TAM) constructs:
- PU: Perceived Usefulness
- EOU: Ease of Use
- Trust: Trust in AI
- ITA: Intention to Adopt

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_physician_demographics(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate demographic data for physicians.
    
    Args:
        n_samples: Number of physician records to generate
        
    Returns:
        DataFrame with physician demographic information
    """
    # Generate unique physician IDs
    physician_ids = [f"PH{str(i).zfill(4)}" for i in range(1, n_samples + 1)]
    
    # Generate ages (25-65) with slight skew towards mid-career physicians
    ages = np.random.normal(loc=42, scale=10, size=n_samples)
    ages = np.clip(ages, 25, 65).astype(int)
    
    # Define specialties with realistic distribution
    specialties = ['Radiology', 'Oncology', 'Cardiology', 'General Medicine', 'Surgery']
    specialty_weights = [0.20, 0.15, 0.18, 0.30, 0.17]  # General Medicine most common
    specialty_choices = np.random.choice(
        specialties, 
        size=n_samples, 
        p=specialty_weights
    )
    
    # Define hospital sizes with distribution
    hospital_sizes = ['Small (<200 beds)', 'Medium (200-500 beds)', 'Large (>500 beds)']
    hospital_weights = [0.25, 0.40, 0.35]  # Medium hospitals most common in SG
    hospital_choices = np.random.choice(
        hospital_sizes, 
        size=n_samples, 
        p=hospital_weights
    )
    
    # Years of experience (correlated with age)
    years_experience = (ages - 25) + np.random.randint(-2, 3, size=n_samples)
    years_experience = np.clip(years_experience, 0, 40)
    
    return pd.DataFrame({
        'physician_id': physician_ids,
        'age': ages,
        'years_experience': years_experience,
        'specialty': specialty_choices,
        'hospital_size': hospital_choices
    })


def generate_tam_scores(demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Technology Acceptance Model (TAM) scores with realistic correlations.
    
    Constructs:
    - PU (Perceived Usefulness): Mean 3.5, independent
    - EOU (Ease of Use): Mean 3.2, negative correlation with age (-0.3)
    - Trust: Mean 3.0, slight positive correlation with EOU
    - ITA (Intention to Adopt): Regressed on PU + EOU + Trust
    
    Args:
        demographics: DataFrame with physician demographics
        
    Returns:
        DataFrame with TAM scores added
    """
    n_samples = len(demographics)
    df = demographics.copy()
    
    # Normalize age for correlation calculations
    age_normalized = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # Generate PU Score (Perceived Usefulness)
    # Mean: 3.5, SD: 0.8, independent of demographics
    pu_base = np.random.normal(loc=3.5, scale=0.8, size=n_samples)
    
    # Add specialty effect (Radiology and Oncology slightly higher PU)
    specialty_pu_effect = df['specialty'].map({
        'Radiology': 0.3,
        'Oncology': 0.2,
        'Cardiology': 0.1,
        'General Medicine': 0.0,
        'Surgery': 0.05
    })
    pu_scores = pu_base + specialty_pu_effect
    pu_scores = np.clip(pu_scores, 1, 5)
    
    # Generate EOU Score (Ease of Use)
    # Mean: 3.2, SD: 0.7, negative correlation with age (r = -0.3)
    eou_base = np.random.normal(loc=3.2, scale=0.7, size=n_samples)
    # Add age effect (older physicians find it harder to use)
    age_effect = -0.3 * age_normalized * 0.7  # Scale by SD
    eou_scores = eou_base + age_effect
    eou_scores = np.clip(eou_scores, 1, 5)
    
    # Generate Trust Score
    # Mean: 3.0, SD: 0.9, slight correlation with EOU (r = 0.2)
    trust_base = np.random.normal(loc=3.0, scale=0.9, size=n_samples)
    # Add EOU effect
    eou_normalized = (eou_scores - eou_scores.mean()) / eou_scores.std()
    eou_trust_effect = 0.2 * eou_normalized * 0.9
    trust_scores = trust_base + eou_trust_effect
    
    # Add hospital size effect (larger hospitals have more AI exposure)
    hospital_trust_effect = df['hospital_size'].map({
        'Small (<200 beds)': -0.15,
        'Medium (200-500 beds)': 0.0,
        'Large (>500 beds)': 0.2
    })
    trust_scores = trust_scores + hospital_trust_effect
    trust_scores = np.clip(trust_scores, 1, 5)
    
    # Generate ITA Score (Intention to Adopt)
    # Regressed on PU + EOU + Trust with coefficients
    # ITA = 0.5 + 0.35*PU + 0.25*EOU + 0.30*Trust + noise
    noise = np.random.normal(loc=0, scale=0.4, size=n_samples)
    ita_scores = (
        0.5 + 
        0.35 * pu_scores + 
        0.25 * eou_scores + 
        0.30 * trust_scores + 
        noise
    )
    ita_scores = np.clip(ita_scores, 1, 5)
    
    # Add scores to dataframe
    df['pu_score'] = np.round(pu_scores, 2)
    df['eou_score'] = np.round(eou_scores, 2)
    df['trust_score'] = np.round(trust_scores, 2)
    df['ita_score'] = np.round(ita_scores, 2)
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for analysis.
    
    Args:
        df: DataFrame with base data
        
    Returns:
        DataFrame with additional derived features
    """
    # Binary adoption intention (ITA > 3.5)
    df['high_adoption_intent'] = (df['ita_score'] > 3.5).astype(int)
    
    # Age group categorization
    df['age_group'] = pd.cut(
        df['age'],
        bins=[24, 35, 45, 55, 66],
        labels=['25-35', '36-45', '46-55', '56-65']
    )
    
    # Experience level
    df['experience_level'] = pd.cut(
        df['years_experience'],
        bins=[-1, 5, 15, 25, 41],
        labels=['Junior (0-5)', 'Mid (6-15)', 'Senior (16-25)', 'Expert (26+)']
    )
    
    # Overall TAM score (average of PU, EOU, Trust)
    df['overall_tam_score'] = np.round(
        (df['pu_score'] + df['eou_score'] + df['trust_score']) / 3, 2
    )
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate generated data meets specifications.
    
    Args:
        df: Generated DataFrame
        
    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    print("\n" + "="*50)
    print("DATA VALIDATION REPORT")
    print("="*50)
    
    # Check sample size
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    print(f"✓ Sample size: {len(df)} rows")
    
    # Check age range
    assert df['age'].min() >= 25, "Age minimum below 25"
    assert df['age'].max() <= 65, "Age maximum above 65"
    print(f"✓ Age range: {df['age'].min()} - {df['age'].max()}")
    
    # Check score ranges (1-5)
    for col in ['pu_score', 'eou_score', 'trust_score', 'ita_score']:
        assert df[col].min() >= 1, f"{col} minimum below 1"
        assert df[col].max() <= 5, f"{col} maximum above 5"
    print("✓ All scores within 1-5 range")
    
    # Check means (approximate)
    print(f"\nScore Statistics:")
    print(f"  PU Score:    Mean = {df['pu_score'].mean():.2f} (target: ~3.5)")
    print(f"  EOU Score:   Mean = {df['eou_score'].mean():.2f} (target: ~3.2)")
    print(f"  Trust Score: Mean = {df['trust_score'].mean():.2f} (target: ~3.0)")
    print(f"  ITA Score:   Mean = {df['ita_score'].mean():.2f} (target: ~3.4)")
    
    # Check EOU-Age correlation
    eou_age_corr = df['eou_score'].corr(df['age'])
    print(f"\nCorrelations:")
    print(f"  EOU-Age correlation: {eou_age_corr:.3f} (target: ~-0.3)")
    
    # Check ITA correlations with predictors
    print(f"  ITA-PU correlation:    {df['ita_score'].corr(df['pu_score']):.3f}")
    print(f"  ITA-EOU correlation:   {df['ita_score'].corr(df['eou_score']):.3f}")
    print(f"  ITA-Trust correlation: {df['ita_score'].corr(df['trust_score']):.3f}")
    
    print("\n✓ All validations passed!")
    return True


def main():
    """
    Main function to generate and save synthetic adoption data.
    """
    print("="*50)
    print("AI DIAGNOSTIC TOOL ADOPTION DATA GENERATOR")
    print("Singapore Hospitals Study - Synthetic Data")
    print("="*50)
    
    # Generate data
    print("\n[1/4] Generating physician demographics...")
    demographics = generate_physician_demographics(n_samples=500)
    
    print("[2/4] Generating TAM scores with correlations...")
    data_with_scores = generate_tam_scores(demographics)
    
    print("[3/4] Adding derived features...")
    final_data = add_derived_features(data_with_scores)
    
    # Validate data
    print("[4/4] Validating generated data...")
    validate_data(final_data)
    
    # Save to CSV
    output_file = 'adoption_data.csv'
    final_data.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to '{output_file}'")
    
    # Display sample
    print("\n" + "="*50)
    print("SAMPLE DATA (First 5 rows)")
    print("="*50)
    print(final_data.head().to_string())
    
    # Display column info
    print("\n" + "="*50)
    print("COLUMN INFORMATION")
    print("="*50)
    print(f"Total columns: {len(final_data.columns)}")
    print(f"Columns: {', '.join(final_data.columns.tolist())}")
    
    return final_data


if __name__ == "__main__":
    data = main()