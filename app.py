import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import urllib.request

# Set page config
st.set_page_config(
    page_title="NFL Betting Model Dashboard", 
    page_icon="🏈", 
    layout="wide"
)

# Google Drive configuration
GOOGLE_DRIVE_FILE_ID = "1vfDVrri9XcIb39F6Ph9vfp_lC2fqAV8K"
DATABASE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm=t"

def get_database_path():
    """Download database from Google Drive if not exists locally"""
    
    # Try local path first (for development)
    local_dev_path = os.path.expanduser("~/Desktop/nfl_app/NFL_Betting_Data_New.db")
    if os.path.exists(local_dev_path):
        return local_dev_path
    
    # Production path (Streamlit Cloud)
    local_db_path = "NFL_Betting_Data_New.db"
    
    # Check if database already downloaded
    if os.path.exists(local_db_path):
        return local_db_path
    
    # Download from Google Drive
    try:
        with st.spinner("🔄 Loading database for the first time (may take 30-60 seconds)..."):
            st.info("📡 Downloading database from Google Drive...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, int((downloaded * 100) / total_size))
                    progress_bar.progress(percent / 100)
                    
                    # Convert to MB for display
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    status_text.text(f"Downloaded: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({percent}%)")
            
            urllib.request.urlretrieve(DATABASE_URL, local_db_path, show_progress)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        st.success("✅ Database loaded successfully!")
        return local_db_path
        
    except Exception as e:
        st.error(f"❌ Error downloading database: {e}")
        st.info("💡 Please check that the Google Drive file is publicly accessible")
        return None

# Get database path
DB_PATH = get_database_path()

@st.cache_data
def load_base_data():
    """Load the base team strength and game data"""
    if not DB_PATH or not os.path.exists(DB_PATH):
        st.error("Database not available")
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = """
        SELECT s.season, s.week, s.gameday,
               s.home_team, s.away_team, 
               s.home_score, s.away_score,
               s.away_moneyline, s.home_moneyline,
               CASE WHEN s.home_score > s.away_score THEN 1 ELSE 0 END as home_won,
               CASE WHEN s.away_score > s.home_score THEN 1 ELSE 0 END as away_won,
               ts_home.base_team_strength as home_base_strength,
               ts_away.base_team_strength as away_base_strength,
               ts_home.coach_score as home_coach_score,
               ts_away.coach_score as away_coach_score,
               ts_home.qbscore as home_qb, ts_away.qbscore as away_qb,
               ts_home.olinescore as home_oline, ts_away.olinescore as away_oline,
               ts_home.dlinescore as home_dline, ts_away.dlinescore as away_dline,
               ts_home.secondaryscore as home_secondary, ts_away.secondaryscore as away_secondary,
               ts_home.lbscore as home_lb, ts_away.lbscore as away_lb,
               ts_home.wrtescore as home_wrte, ts_away.wrtescore as away_wrte,
               ts_home.rbscore as home_rb, ts_away.rbscore as away_rb,
               s.spread_line
        FROM schedules s
        INNER JOIN team_strength ts_home ON s.home_team = ts_home.team 
            AND s.season = ts_home.season AND s.week = ts_home.week
        INNER JOIN team_strength ts_away ON s.away_team = ts_away.team 
            AND s.season = ts_away.season AND s.week = ts_away.week
        WHERE s.game_type = 'REG' 
        AND s.season IN (2023, 2024)
        AND s.home_score IS NOT NULL 
        AND s.away_score IS NOT NULL
        AND s.away_moneyline IS NOT NULL
        AND s.home_moneyline IS NOT NULL
        AND ts_home.base_team_strength IS NOT NULL
        AND ts_away.base_team_strength IS NOT NULL
        AND ts_home.coach_score IS NOT NULL
        AND ts_away.coach_score IS NOT NULL
        ORDER BY s.season, s.week, s.gameday
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_betting_data():
    """Load basic game and betting data for favorite model"""
    if not DB_PATH or not os.path.exists(DB_PATH):
        st.error("Database not available")
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = """
        SELECT s.season, s.week, s.gameday,
               s.home_team, s.away_team, 
               s.home_score, s.away_score,
               s.away_moneyline, s.home_moneyline,
               CASE WHEN s.home_score > s.away_score THEN 1 ELSE 0 END as home_won,
               CASE WHEN s.away_score > s.home_score THEN 1 ELSE 0 END as away_won,
               s.spread_line
        FROM schedules s
        WHERE s.game_type = 'REG' 
        AND s.season IN (2020, 2021, 2022, 2023, 2024)
        AND s.home_score IS NOT NULL 
        AND s.away_score IS NOT NULL
        AND s.away_moneyline IS NOT NULL
        AND s.home_moneyline IS NOT NULL
        ORDER BY s.season, s.week, s.gameday
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_talent_adjusted_strength(row, position_weights, coach_factor):
    """Calculate talent-adjusted strength for a team"""
    home_talent = (
        row['home_qb'] * position_weights['QB'] +
        row['home_oline'] * position_weights['OLine'] +
        row['home_dline'] * position_weights['DLine'] +
        row['home_secondary'] * position_weights['Secondary'] +
        row['home_lb'] * position_weights['LB'] +
        row['home_wrte'] * position_weights['WRTE'] +
        row['home_rb'] * position_weights['RB']
    )
    
    away_talent = (
        row['away_qb'] * position_weights['QB'] +
        row['away_oline'] * position_weights['OLine'] +
        row['away_dline'] * position_weights['DLine'] +
        row['away_secondary'] * position_weights['Secondary'] +
        row['away_lb'] * position_weights['LB'] +
        row['away_wrte'] * position_weights['WRTE'] +
        row['away_rb'] * position_weights['RB']
    )
    
    home_coach_adjusted = row['home_coach_score'] * home_talent
    away_coach_adjusted = row['away_coach_score'] * away_talent
    
    home_final = row['home_base_strength'] + (home_coach_adjusted / coach_factor)
    away_final = row['away_base_strength'] + (away_coach_adjusted / coach_factor)
    
    return home_final, away_final

def analyze_betting_performance(df, position_weights, coach_factor, betting_strategy, odds_min, odds_max, seasons, week_range):
    """Analyze betting performance with given parameters"""
    
    df_filtered = df[
        (df['season'].isin(seasons)) & 
        (df['week'] >= week_range[0]) & 
        (df['week'] <= week_range[1])
    ].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame(), {}
    
    talent_strengths = df_filtered.apply(
        lambda row: calculate_talent_adjusted_strength(row, position_weights, coach_factor), 
        axis=1
    )
    
    df_filtered['home_talent_strength'] = [x[0] for x in talent_strengths]
    df_filtered['away_talent_strength'] = [x[1] for x in talent_strengths]
    df_filtered['strength_differential'] = abs(df_filtered['home_talent_strength'] - df_filtered['away_talent_strength'])
    
    bets = []
    
    for _, game in df_filtered.iterrows():
        home_strength = game['home_talent_strength']
        away_strength = game['away_talent_strength']
        home_ml = game['home_moneyline']
        away_ml = game['away_moneyline']
        home_won = game['home_won']
        away_won = game['away_won']
        
        bet_placed = False
        bet_team = None
        bet_moneyline = None
        model_won = None
        bet_type = None
        
        if betting_strategy == "Favorites Only":
            if home_strength > away_strength:
                if home_ml < 0 and odds_min <= abs(home_ml) <= odds_max:
                    bet_placed = True
                    bet_team = game['home_team']
                    bet_moneyline = home_ml
                    model_won = home_won
                    bet_type = "Favorite (Home)"
            else:
                if away_ml < 0 and odds_min <= abs(away_ml) <= odds_max:
                    bet_placed = True
                    bet_team = game['away_team']
                    bet_moneyline = away_ml
                    model_won = away_won
                    bet_type = "Favorite (Away)"
                    
        elif betting_strategy == "Home Teams Only":
            if home_strength > away_strength:
                if home_ml < 0 and odds_min <= abs(home_ml) <= odds_max:
                    bet_moneyline = home_ml
                    bet_type = "Home Favorite"
                elif home_ml > 0 and odds_min <= home_ml <= odds_max:
                    bet_moneyline = home_ml
                    bet_type = "Home Underdog"
                else:
                    continue
                    
                bet_placed = True
                bet_team = game['home_team']
                model_won = home_won
                
        elif betting_strategy == "Away Teams Only":
            if away_strength > home_strength:
                if away_ml < 0 and odds_min <= abs(away_ml) <= odds_max:
                    bet_moneyline = away_ml
                    bet_type = "Away Favorite"
                elif away_ml > 0 and odds_min <= away_ml <= odds_max:
                    bet_moneyline = away_ml
                    bet_type = "Away Underdog"
                else:
                    continue
                    
                bet_placed = True
                bet_team = game['away_team']
                model_won = away_won
                
        elif betting_strategy == "All Model Picks":
            if home_strength > away_strength:
                if home_ml < 0 and odds_min <= abs(home_ml) <= odds_max:
                    bet_moneyline = home_ml
                    bet_type = "Model Pick (Home Fav)"
                elif home_ml > 0 and odds_min <= home_ml <= odds_max:
                    bet_moneyline = home_ml
                    bet_type = "Model Pick (Home Dog)"
                else:
                    continue
                bet_team = game['home_team']
                model_won = home_won
            else:
                if away_ml < 0 and odds_min <= abs(away_ml) <= odds_max:
                    bet_moneyline = away_ml
                    bet_type = "Model Pick (Away Fav)"
                elif away_ml > 0 and odds_min <= away_ml <= odds_max:
                    bet_moneyline = away_ml
                    bet_type = "Model Pick (Away Dog)"
                else:
                    continue
                bet_team = game['away_team']
                model_won = away_won
            bet_placed = True
        
        if bet_placed:
            if bet_moneyline < 0:
                bet_amount = abs(bet_moneyline)
                profit_if_win = 100
            else:
                bet_amount = 100
                profit_if_win = bet_moneyline
            
            actual_profit = profit_if_win if model_won else -bet_amount
            
            if home_won:
                winner = game['home_team']
            else:
                winner = game['away_team']
            
            bets.append({
                'season': game['season'],
                'week': game['week'],
                'date': game['gameday'],
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'bet_team': bet_team,
                'bet_type': bet_type,
                'home_strength': round(home_strength, 3),
                'away_strength': round(away_strength, 3),
                'strength_differential': round(game['strength_differential'], 3),
                'bet_moneyline': bet_moneyline,
                'home_moneyline': home_ml,
                'away_moneyline': away_ml,
                'bet_amount': bet_amount,
                'potential_profit': profit_if_win,
                'winner': winner,
                'final_score': f"{int(game['home_score'])}-{int(game['away_score'])}",
                'model_won': model_won,
                'actual_profit': actual_profit
            })
    
    bets_df = pd.DataFrame(bets)
    
    if len(bets_df) == 0:
        return bets_df, {}
    
    total_bets = len(bets_df)
    total_wins = bets_df['model_won'].sum()
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    total_profit = bets_df['actual_profit'].sum()
    total_amount_bet = bets_df['bet_amount'].sum()
    roi = total_profit / total_amount_bet if total_amount_bet > 0 else 0
    
    summary = {
        'total_bets': total_bets,
        'total_wins': int(total_wins),
        'total_losses': total_bets - int(total_wins),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_amount_bet': total_amount_bet,
        'roi': roi,
        'avg_bet_size': total_amount_bet / total_bets if total_bets > 0 else 0
    }
    
    return bets_df, summary

def analyze_favorite_betting(df, odds_min, odds_max, seasons, week_range, bet_size):
    """Analyze betting performance by always betting the moneyline favorite"""
    
    df_filtered = df[
        (df['season'].isin(seasons)) & 
        (df['week'] >= week_range[0]) & 
        (df['week'] <= week_range[1])
    ].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame(), {}
    
    df_filtered['home_is_favorite'] = df_filtered['home_moneyline'] < df_filtered['away_moneyline']
    df_filtered['away_is_favorite'] = df_filtered['away_moneyline'] < df_filtered['home_moneyline']
    
    df_filtered['favorite_team'] = np.where(
        df_filtered['home_is_favorite'], 
        df_filtered['home_team'], 
        df_filtered['away_team']
    )
    df_filtered['favorite_odds'] = np.where(
        df_filtered['home_is_favorite'], 
        df_filtered['home_moneyline'], 
        df_filtered['away_moneyline']
    )
    df_filtered['underdog_team'] = np.where(
        df_filtered['home_is_favorite'], 
        df_filtered['away_team'], 
        df_filtered['home_team']
    )
    df_filtered['underdog_odds'] = np.where(
        df_filtered['home_is_favorite'], 
        df_filtered['away_moneyline'], 
        df_filtered['home_moneyline']
    )
    df_filtered['favorite_won'] = np.where(
        df_filtered['home_is_favorite'], 
        df_filtered['home_won'], 
        df_filtered['away_won']
    )
    
    df_filtered = df_filtered[
        (abs(df_filtered['favorite_odds']) >= odds_min) & 
        (abs(df_filtered['favorite_odds']) <= odds_max)
    ]
    
    bets = []
    
    for _, game in df_filtered.iterrows():
        favorite_odds = game['favorite_odds']
        favorite_won = game['favorite_won']
        
        if favorite_odds < 0:
            bet_amount = abs(favorite_odds) * (bet_size / 100)
            profit_if_win = bet_size
        else:
            bet_amount = bet_size
            profit_if_win = favorite_odds * (bet_size / 100)
        
        actual_profit = profit_if_win if favorite_won else -bet_amount
        
        if game['home_won']:
            winner = game['home_team']
        else:
            winner = game['away_team']
        
        bets.append({
            'season': game['season'],
            'week': game['week'],
            'date': game['gameday'],
            'matchup': f"{game['away_team']} @ {game['home_team']}",
            'favorite_team': game['favorite_team'],
            'underdog_team': game['underdog_team'],
            'favorite_odds': favorite_odds,
            'underdog_odds': game['underdog_odds'],
            'spread_line': game['spread_line'],
            'bet_amount': bet_amount,
            'potential_profit': profit_if_win,
            'winner': winner,
            'final_score': f"{int(game['home_score'])}-{int(game['away_score'])}",
            'favorite_won': favorite_won,
            'actual_profit': actual_profit,
            'favorite_location': 'Home' if game['home_is_favorite'] else 'Away'
        })
    
    bets_df = pd.DataFrame(bets)
    
    if len(bets_df) == 0:
        return bets_df, {}
    
    total_bets = len(bets_df)
    total_wins = bets_df['favorite_won'].sum()
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    total_profit = bets_df['actual_profit'].sum()
    total_amount_bet = bets_df['bet_amount'].sum()
    roi = total_profit / total_amount_bet if total_amount_bet > 0 else 0
    
    summary = {
        'total_bets': total_bets,
        'total_wins': int(total_wins),
        'total_losses': total_bets - int(total_wins),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_amount_bet': total_amount_bet,
        'roi': roi,
        'avg_bet_size': total_amount_bet / total_bets if total_bets > 0 else 0,
        'avg_favorite_odds': bets_df['favorite_odds'].mean()
    }
    
    return bets_df, summary

def render_complex_model_tab():
    """Render the complex betting model tab"""
    st.header("🧠 Complex Team Strength Model")
    st.markdown("Advanced model using team strength, coaching, and talent metrics")
    
    with st.spinner("Loading team strength data..."):
        df = load_base_data()
    
    if df.empty:
        st.error("Failed to load data. Please check your database connection.")
        return
    
    st.success(f"Loaded {len(df)} games from 2023-2024 seasons")
    
    st.sidebar.header("Model Configuration")
    st.sidebar.subheader("Position Weights")
    
    default_weights = {
        'QB': 0.30,
        'OLine': 0.20,
        'DLine': 0.15,
        'Secondary': 0.15,
        'LB': 0.075,
        'WRTE': 0.075,
        'RB': 0.05
    }
    
    for position, default_val in default_weights.items():
        key = f"weight_{position}"
        if key not in st.session_state:
            st.session_state[key] = default_val
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    if col1.button("Current"):
        for key, value in default_weights.items():
            st.session_state[f"weight_{key}"] = value
    
    if col2.button("QB Heavy"):
        qb_heavy = {'QB': 0.45, 'OLine': 0.20, 'DLine': 0.125, 'Secondary': 0.125, 'LB': 0.05, 'WRTE': 0.05, 'RB': 0.0}
        for key, value in qb_heavy.items():
            st.session_state[f"weight_{key}"] = value
    
    if col3.button("Defense"):
        defense_heavy = {'QB': 0.20, 'OLine': 0.15, 'DLine': 0.25, 'Secondary': 0.25, 'LB': 0.10, 'WRTE': 0.05, 'RB': 0.0}
        for key, value in defense_heavy.items():
            st.session_state[f"weight_{key}"] = value
    
    position_weights = {}
    for position in default_weights.keys():
        position_weights[position] = st.sidebar.slider(
            f"{position} Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state[f"weight_{position}"],
            step=0.025,
            key=f"complex_{position}"
        )
    
    total_weight = sum(position_weights.values())
    if abs(total_weight - 1.0) > 0.001:
        st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.3f}, not 1.0")
    else:
        st.sidebar.success(f"✅ Weights sum to {total_weight:.3f}")
    
    st.sidebar.subheader("Talent Adjustment Settings")
    coach_factor = st.sidebar.slider(
        "Coach × Talent Scaling Factor",
        min_value=100.0,
        max_value=2000.0,
        value=250.0,
        step=25.0,
        key="complex_coach_factor"
    )
    
    st.sidebar.subheader("Betting Strategy")
    betting_strategy = st.sidebar.selectbox(
        "Strategy",
        ["Favorites Only", "Home Teams Only", "Away Teams Only", "All Model Picks"],
        key="complex_strategy"
    )
    
    st.sidebar.subheader("Odds Filter")
    odds_min = st.sidebar.slider("Min Odds (absolute value)", 100, 500, 100, step=10, key="complex_odds_min")
    odds_max = st.sidebar.slider("Max Odds (absolute value)", 100, 500, 350, step=10, key="complex_odds_max")
    
    st.sidebar.subheader("Analysis Period")
    seasons = st.sidebar.multiselect("Seasons", [2023, 2024], default=[2023, 2024], key="complex_seasons")
    week_range = st.sidebar.slider("Week Range", 1, 18, (1, 18), key="complex_weeks")
    
    if st.sidebar.button("🔄 Run Complex Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            bets_df, summary = analyze_betting_performance(
                df, position_weights, coach_factor, betting_strategy, 
                odds_min, odds_max, seasons, week_range
            )
            
            st.session_state['complex_bets_df'] = bets_df
            st.session_state['complex_summary'] = summary
            st.session_state['complex_analysis_complete'] = True
    
    if st.session_state.get('complex_analysis_complete', False):
        bets_df = st.session_state['complex_bets_df']
        summary = st.session_state['complex_summary']
        
        if len(bets_df) == 0:
            st.warning("No bets found with current settings. Try adjusting your filters.")
            return
        
        st.subheader("📊 Summary Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bets", summary['total_bets'])
            st.metric("Wins", summary['total_wins'])
        
        with col2:
            st.metric("Win Rate", f"{summary['win_rate']:.1%}")
            st.metric("Losses", summary['total_losses'])
        
        with col3:
            st.metric("Total Profit", f"${summary['total_profit']:,.0f}")
            st.metric("Amount Bet", f"${summary['total_amount_bet']:,.0f}")
        
        with col4:
            st.metric("ROI", f"{summary['roi']:+.1%}")
            st.metric("Avg Favorite Odds", f"{summary['avg_favorite_odds']:.0f}")

def main():
    st.title("🏈 NFL Betting Model Dashboard")
    st.markdown("Compare complex team strength model vs simple favorite betting")
    
    # Show database status
    if DB_PATH and os.path.exists(DB_PATH):
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
        st.sidebar.success(f"📊 Database loaded ({db_size:.0f}MB)")
    else:
        st.sidebar.error("❌ Database not loaded")
        st.stop()
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🧠 Complex Model", "⭐ Favorite Model"])
    
    with tab1:
        render_complex_model_tab()
    
    with tab2:
        render_favorite_model_tab()

if __name__ == "__main__":
    main()
