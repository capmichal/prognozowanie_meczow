"""
Smart-Bet SaaS Platform - MVP Interface
=======================================
A Streamlit-based web application for algorithmic sports predictions
using Expected Goals (xG) data and advanced statistical modeling.

Author: Master's Thesis in Business Engineering
Date: January 2026
"""

import streamlit as st
import pandas as pd
from pipeline import BettingEngine


# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Smart-Bet | xG Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================
# Helper Functions
# ===========================
@st.cache_resource
def initialize_engine():
    """
    Initialize and prepare the BettingEngine.
    Uses caching to avoid reloading data on every interaction.
    
    Returns:
        BettingEngine: Initialized engine with prepared data
    """
    with st.spinner("Loading data and initializing prediction engine..."):
        engine = BettingEngine(
            path_results="datasets/premier_league_20-21.csv",
            path_shots="datasets/shots_premier_league_20-21.csv",
            window_size=5
        )
        engine.prepare_data()
    return engine


def get_team_list(engine):
    """
    Extract unique team names from the dataset.
    
    Args:
        engine (BettingEngine): Initialized engine instance
        
    Returns:
        list: Sorted list of unique team names
    """
    home_teams = engine.df['HomeTeam'].unique()
    away_teams = engine.df['AwayTeam'].unique()
    all_teams = sorted(set(list(home_teams) + list(away_teams)))
    return all_teams


# ===========================
# Main Application
# ===========================
def main():
    """
    Main application logic for the Smart-Bet platform.
    """
    
    # Header Section
    st.title("⚽ Smart-Bet | Algorithmic Football Predictions")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <p style='margin: 0; color: #31333F;'>
            <strong>Advanced xG-based predictions</strong> using rolling form analysis, 
            Attack Strength (AS), and Defense Strength (DS) metrics relative to league averages.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the engine
    try:
        engine = initialize_engine()
        teams = get_team_list(engine)
        
        st.success(f"✅ Engine initialized successfully! Loaded data for {len(teams)} teams.")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize engine: {str(e)}")
        st.stop()
    
    # Sidebar - Information Panel
    with st.sidebar:
        st.header("📊 About the Model")
        st.markdown("""
        **Data Source:**  
        Premier League 2020-21 Season
        
        **Key Metrics:**
        - **AS (Attack Strength):** Team's offensive capability relative to league average
        - **DS (Defense Strength):** Team's defensive vulnerability relative to league average
        - **Rolling Form:** Last 5 matches performance
        
        **Prediction Formula:**
        ```
        Home xG = Home_AS × Away_DS × League_Avg_Home_xG
        Away xG = Away_AS × Home_DS × League_Avg_Away_xG
        ```
        """)
        
        st.divider()
        st.caption("Smart-Bet v1.0 | Master's Thesis Project")
    
    # Main Content Area
    st.header("🔮 Match Prediction")
    
    # Team Selection Section
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "🏠 Home Team",
            options=teams,
            index=0,
            help="Select the team playing at home"
        )
    
    with col2:
        away_team = st.selectbox(
            "✈️ Away Team",
            options=teams,
            index=1 if len(teams) > 1 else 0,
            help="Select the team playing away"
        )
    
    # Validation
    if home_team == away_team:
        st.warning("⚠️ Please select two different teams.")
        st.stop()
    
    # Analyze Button
    st.divider()
    analyze_button = st.button(
        "🚀 Analyze Match",
        type="primary",
        use_container_width=True
    )
    
    # Prediction Logic
    if analyze_button:
        try:
            with st.spinner("Analyzing match data..."):
                prediction = engine.get_prediction(home_team, away_team)
            
            # Display Results
            st.success("✅ Analysis complete!")
            st.divider()
            
            # Match Header
            st.subheader(f"📋 {prediction['home_team']} vs {prediction['away_team']}")
            
            # Metrics Display
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric(
                    label=f"🏠 {prediction['home_team']} (Home)",
                    value=f"{prediction['home_xg']} xG",
                    delta="Favorite" if prediction['favorite'] == home_team else None,
                    delta_color="normal" if prediction['favorite'] == home_team else "off"
                )
            
            with col2:
                # Visual Separator
                st.markdown("""
                <div style='text-align: center; padding-top: 2rem;'>
                    <h2 style='color: #31333F; margin: 0;'>VS</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric(
                    label=f"✈️ {prediction['away_team']} (Away)",
                    value=f"{prediction['away_xg']} xG",
                    delta="Favorite" if prediction['favorite'] == away_team else None,
                    delta_color="normal" if prediction['favorite'] == away_team else "off"
                )
            
            # Statistical Favorite
            st.divider()
            if prediction['favorite'] == "Draw":
                st.info("📊 **Statistical Outlook:** Even match - Both teams have similar expected performance")
            else:
                xg_difference = abs(prediction['home_xg'] - prediction['away_xg'])
                st.info(f"📊 **Statistical Favorite:** {prediction['favorite']} (+{xg_difference:.2f} xG advantage)")
            
            # Additional Insights
            with st.expander("📈 View Detailed Insights"):
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown(f"""
                    **{prediction['home_team']} (Home Team)**
                    - Predicted xG: {prediction['home_xg']}
                    - Home Advantage: Yes
                    - League Avg Home xG: {engine.league_avg_h_xg:.2f}
                    """)
                
                with insight_col2:
                    st.markdown(f"""
                    **{prediction['away_team']} (Away Team)**
                    - Predicted xG: {prediction['away_xg']}
                    - Home Advantage: No
                    - League Avg Away xG: {engine.league_avg_a_xg:.2f}
                    """)
                
                st.caption("*xG values are calculated based on recent form and strength metrics.*")
        
        except ValueError as e:
            st.error(f"❌ Prediction Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error occurred: {str(e)}")
    
    # Footer
    st.divider()
    st.caption("""
    **Disclaimer:** This tool provides statistical predictions based on historical data. 
    Actual match outcomes may vary due to numerous unpredictable factors.
    """)


# ===========================
# Application Entry Point
# ===========================
if __name__ == "__main__":
    main()
