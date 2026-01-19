import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Football Crowd Dashboard", layout="wide", initial_sidebar_state="expanded")

DATA_PATH = "data/Matches_clean_final.csv"
TOP_LEAGUES = ["E0", "SP1", "I1", "D1", "F1"]

# -------------------------
# Load (cached)
# -------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # types
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    for c in ["Attendance", "StadiumCapacity", "HomeElo", "AwayElo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # base cleaning
    df = df[(df["StadiumCapacity"] > 0) & df["Attendance"].notna() & df["HomeElo"].notna() & df["AwayElo"].notna()].copy()
    df["AttendanceRate"] = (df["Attendance"] / df["StadiumCapacity"]).clip(0, 1)

    # home win
    df["HomeWin"] = (df["FTResult"] == "H").astype(int)

    # elo diff
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["GameState"] = pd.cut(
        df["EloDiff"],
        bins=[-1e9, -50, 50, 1e9],
        labels=["Underdog", "Even", "Favorite"]
    )

    return df

df = load_data(DATA_PATH)
df = df[df["Division"].isin(TOP_LEAGUES)].copy()

# =========================================================
# Navigation / Sidebar
# =========================================================
page = st.sidebar.radio(
    "📊 Navigation",
    ["🏠 Home", "📈 Analysis", "🔍 Deep Dive", "⚽ Elo vs Attendance"],
    label_visibility="collapsed"
)

# =========================================================
# PAGE: HOME
# =========================================================
if page == "🏠 Home":
    st.title("⚽ Crowd Impact on Football Performance")
    st.markdown("""
    ---
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Matches", f"{len(df):,}")
    with col2:
        st.metric("🏆 Leagues Analyzed", f"{df['Division'].nunique()}")
    with col3:
        st.metric("📅 Seasons Covered", f"{int(df['Season'].min())} – {int(df['Season'].max())}")
    
    st.markdown("""
    ---
    
    ## 🎯 Main Insight
    
    **Does home crowd really matter?** This dashboard quantifies the effect of stadium attendance on home team performance, 
    controlling for team strength (Elo rating).
    
    ### Key Findings:
    
    """)
    
    # Quick stats
    ghost_matches = df[df["AttendanceRate"] <= 0.30]
    crowd_matches = df[df["AttendanceRate"] >= 0.70]
    
    if not ghost_matches.empty and not crowd_matches.empty:
        ghost_winpct = 100 * ghost_matches["HomeWin"].mean()
        crowd_winpct = 100 * crowd_matches["HomeWin"].mean()
        effect = crowd_winpct - ghost_winpct
        
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.metric("👻 Low Crowd Win %", f"{ghost_winpct:.1f}%")
        with col_k2:
            st.metric("🏟️ High Crowd Win %", f"{crowd_winpct:.1f}%")
        with col_k3:
            st.metric("📊 Crowd Effect", f"+{effect:.1f} pp", delta=f"{effect:+.1f} pp")
    
    st.markdown("""
    ---
    
    ## 📋 How to Use This Dashboard
    
    1. **📈 Analysis Tab**: View league-wide trends and upset rates by match context
       - Compare home win percentages in low vs high attendance conditions
       - See how crowd effect varies by favorite strength
    
    2. **🔍 Deep Dive Tab**: Team-level analysis with individual controls
       - Analyze residual crowd effect after controlling for team strength (Elo)
       - Highlight specific teams and see their crowd sensitivity
    
    3. **⚙️ Customize**: Use sidebar filters to adjust:
       - Attendance thresholds (what counts as "ghost" vs "crowd")
       - Season range and league selection
       - Minimum game thresholds for reliability
    
    ---
    
    ## 💡 Methodology
    
    - **Team Strength**: Measured using Elo ratings
    - **Crowd Classification**: Attendance rate thresholds (customizable)
    - **Upset Analysis**: Based on Elo-predicted favorites vs actual outcomes
    - **Filtering**: Minimum games required per category to ensure statistical reliability
    
    ---
    
    """)
    
    st.info("""
    **💬 Interpretation Tip**: 
    
    A positive crowd effect means:
    - Home teams win **more often** with high attendance
    - This could be due to **psychological boost** or **player motivation**
    
    Analyzing "upsets" (when favorites don't win) helps isolate crowd impact from team strength.
    """)

# =========================================================
# PAGE: ANALYSIS
# =========================================================
elif page == "📈 Analysis":
    st.title("Crowd Impact Dashboard")
    st.caption("Analysis of crowd effect on home team performance.")

    # -------------------------
    # Sidebar filters (for Analysis page)
    # -------------------------
    st.sidebar.header("Analysis Page Filters")

    st.sidebar.divider()
    st.sidebar.subheader("Crowd classification")
    fixed_low_analysis = st.sidebar.slider("Ghost threshold (AttendanceRate)", 0.00, 0.60, 0.30, 0.01)
    fixed_high_analysis = st.sidebar.slider("Crowd threshold (AttendanceRate)", 0.40, 1.00, 0.70, 0.01)

    def apply_crowd_status(dfi: pd.DataFrame) -> pd.DataFrame:
        dfo = dfi.copy()
        dfo["CrowdStatus"] = pd.Series(pd.NA, index=dfo.index, dtype="string")
        dfo.loc[dfo["AttendanceRate"] <= fixed_low_analysis, "CrowdStatus"] = "Ghost"
        dfo.loc[dfo["AttendanceRate"] >= fixed_high_analysis, "CrowdStatus"] = "Crowd"

        return dfo[dfo["CrowdStatus"].notna()].copy()

    # =========================================================
    # Dumbbell Plot: Home Win % by League (Ghost vs Crowd)
    # =========================================================
    st.header("Home Win % by League: Low Crowd vs High Crowd")

    col_l, col_r = st.columns([1, 0.001])

    # Filter data for dumbbell plot
    dff = apply_crowd_status(df)

    if not dff.empty:
        # Aggregate: Home win % per league × crowd status
        league_stats = (
            dff.groupby(["Division", "CrowdStatus"], as_index=False)
               .agg(HomeWinPct=("HomeWin", lambda s: 100.0 * s.mean()),
                    Matches=("HomeWin", "size"))
        )

        wide_dumb = league_stats.pivot(index="Division", columns="CrowdStatus", values="HomeWinPct").reset_index()
        wide_dumb = wide_dumb.dropna(subset=["Ghost", "Crowd"]).copy()
        wide_dumb["Effect_pp"] = wide_dumb["Crowd"] - wide_dumb["Ghost"]
        wide_dumb = wide_dumb.sort_values("Effect_pp", ascending=False).reset_index(drop=True)

        # Build dumbbell plot
        fig_dumb = go.Figure()

        # Connecting lines
        for _, row in wide_dumb.iterrows():
            fig_dumb.add_trace(go.Scatter(
                x=[row["Ghost"], row["Crowd"]],
                y=[row["Division"], row["Division"]],
                mode="lines",
                line=dict(width=6, color="rgba(100,100,100,0.3)"),
                showlegend=False,
                hoverinfo="skip"
            ))

        # Ghost points
        fig_dumb.add_trace(go.Scatter(
            x=wide_dumb["Ghost"],
            y=wide_dumb["Division"],
            mode="markers+text",
            text=[f"{v:.1f}%" for v in wide_dumb["Ghost"]],
            textposition="middle left",
            marker=dict(size=12, color="steelblue"),
            name="Ghost (low crowd)",
            customdata=np.stack([wide_dumb["Effect_pp"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Ghost: %{x:.1f}%<br>Effect: %{customdata[0]:.1f} pp<extra></extra>"
        ))

        # Crowd points
        fig_dumb.add_trace(go.Scatter(
            x=wide_dumb["Crowd"],
            y=wide_dumb["Division"],
            mode="markers+text",
            text=[f"{v:.1f}%" for v in wide_dumb["Crowd"]],
            textposition="middle right",
            marker=dict(size=12, color="coral"),
            name="Crowd (high crowd)",
            customdata=np.stack([wide_dumb["Effect_pp"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Crowd: %{x:.1f}%<br>Effect: %{customdata[0]:.1f} pp<extra></extra>"
        ))

        crowd_label = f"Ghost (≤{fixed_low_analysis:.0%}) vs Crowd (≥{fixed_high_analysis:.0%})"

        fig_dumb.update_layout(
            title=f"Home Win % by League: {crowd_label} — sorted by effect",
            xaxis_title="Home Win Percentage (%)",
            yaxis_title="League",
            height=450,
            margin=dict(l=60, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="right", x=1.0),
            hovermode="closest"
        )

        with col_l:
            st.plotly_chart(fig_dumb, use_container_width=True)

        st.divider()

    # =========================================================
    # Supporting View 2: Upset Rate by Favorite Strength
    # =========================================================
    st.header("Upset Rate by Favorite Strength – Ghost vs Crowd")

    # Build favorite / upset definitions
    tmp = dff.copy()
    tmp = tmp[tmp["HomeElo"].notna() & tmp["AwayElo"].notna()].copy()

    tmp["EloDiffAbs"] = (tmp["HomeElo"] - tmp["AwayElo"]).abs()
    
    # Favorite side
    tmp["FavSide"] = np.where(tmp["HomeElo"] >= tmp["AwayElo"], "Home", "Away")
    tmp["FavWin"] = ((tmp["FavSide"] == "Home") & (tmp["FTResult"] == "H")) | ((tmp["FavSide"] == "Away") & (tmp["FTResult"] == "A"))

    # Upset: favorite did not win (includes draw as upset)
    tmp["Upset"] = (~tmp["FavWin"]).astype(int)

    # threshold for being "a favorite game"
    fav_min = st.slider("Min EloDiffAbs to count as 'favorite game'", 0, 140, 50, 10)
    tmp = tmp[tmp["EloDiffAbs"] >= fav_min].copy()

    # bins (editable)
    bins = [fav_min, 150, 300, np.inf]
    labels = ["Small Fav", "Strong Fav", "Huge Fav"]
    tmp["FavStrength"] = pd.cut(tmp["EloDiffAbs"], bins=bins, labels=labels, right=False)
    tmp = tmp[tmp["FavStrength"].notna()].copy()

    if not tmp.empty:
        stats2 = (
            tmp.groupby(["FavStrength", "CrowdStatus"], as_index=False)
               .agg(UpsetRate=("Upset", lambda s: 100.0 * s.mean()),
                    Matches=("Upset", "size"))
        )

        wide2 = stats2.pivot(index="FavStrength", columns="CrowdStatus", values="UpsetRate").reset_index()
        wide2 = wide2.dropna(subset=["Ghost", "Crowd"]).copy()
        
        if not wide2.empty:
            wide2["Effect_pp"] = wide2["Ghost"] - wide2["Crowd"]  # positive = more upsets without crowd
            wide2 = wide2.set_index("FavStrength").reindex(labels).reset_index()
            wide2 = wide2.dropna(subset=["Crowd", "Ghost"])

            # Build dumbbell plot for upset rate
            fig2 = go.Figure()

            for _, r in wide2.iterrows():
                fig2.add_trace(go.Scatter(
                    x=[r["Crowd"], r["Ghost"]],
                    y=[r["FavStrength"], r["FavStrength"]],
                    mode="lines",
                    line=dict(width=7, color="rgba(100,100,100,0.3)"),
                    showlegend=False,
                    hoverinfo="skip"
                ))

            fig2.add_trace(go.Scatter(
                x=wide2["Crowd"],
                y=wide2["FavStrength"],
                mode="markers+text",
                text=[f"{v:.1f}%" for v in wide2["Crowd"]],
                textposition="top center",
                marker=dict(size=12, color="steelblue"),
                name="Crowd",
                customdata=np.stack([wide2["Effect_pp"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>Upset Rate (Crowd): %{x:.1f}%<br>Effect: %{customdata[0]:.1f} pp<extra></extra>"
            ))

            fig2.add_trace(go.Scatter(
                x=wide2["Ghost"],
                y=wide2["FavStrength"],
                mode="markers+text",
                text=[f"{v:.1f}%" for v in wide2["Ghost"]],
                textposition="bottom center",
                marker=dict(size=12, color="coral"),
                name="Ghost",
                customdata=np.stack([wide2["Effect_pp"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>Upset Rate (Ghost): %{x:.1f}%<br>Effect: %{customdata[0]:.1f} pp<extra></extra>"
            ))

            crowd_label = f"Ghost (≤{fixed_low_analysis:.0%}) vs Crowd (≥{fixed_high_analysis:.0%})"

            fig2.update_layout(
                height=380,
                margin=dict(l=60, r=40, t=60, b=40),
                xaxis_title="Upset Rate (%) — Favorite Did NOT Win",
                yaxis_title="Favorite Strength (by EloDiff)",
                title=f"Upset Rate by Favorite Strength: {crowd_label}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                hovermode="closest"
            )

            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Show summary table (Graph 2)"):
                st.dataframe(
                    wide2[["FavStrength", "Crowd", "Ghost", "Effect_pp"]]
                        .rename(columns={"Crowd": "Upset % (Crowd)", "Ghost": "Upset % (Ghost)", "Effect_pp": "Effect (pp)"}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No data available for upset rate analysis after filtering.")
    else:
        st.warning("No favorite games after current EloDiff threshold.")

    st.divider()

    # =========================================================
    # A3: Upsets Radar (Ghost vs Crowd)
    # =========================================================
    st.markdown(
        """
        <style>
        .radar-wrap {
            background: linear-gradient(180deg, #11141a 0%, #0b0d12 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            margin-bottom: 16px;
        }
        .radar-title {
            font-size: 34px;
            font-weight: 800;
            letter-spacing: 0.2px;
            color: #f1f5f9;
            margin: 0 0 6px 0;
        }
        .radar-subtitle {
            color: #9aa4b2;
            font-size: 14.5px;
            margin: 0 0 6px 0;
        }
        </style>
        <div class="radar-wrap">
            <div class="radar-title">The Expanding Web of Upsets</div>
            <div class="radar-subtitle">Ghost vs Crowd — Upset rate by favorite strength</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dff_radar = df.copy()
    dff_radar["CrowdStatus"] = pd.Series(pd.NA, index=dff_radar.index, dtype="string")
    dff_radar.loc[dff_radar["AttendanceRate"] <= fixed_low_analysis, "CrowdStatus"] = "Ghost (Empty)"
    dff_radar.loc[dff_radar["AttendanceRate"] >= fixed_high_analysis, "CrowdStatus"] = "Crowd (Full)"

    dff_radar = dff_radar[dff_radar["CrowdStatus"].notna()].copy()
    dff_radar = dff_radar[dff_radar["EloDiff"] > 0].copy()

    dff_radar["EloBin"] = pd.cut(
        dff_radar["EloDiff"],
        bins=[50, 150, 300, 1000],
        labels=["Small Fav", "Strong Fav", "Huge Fav"]
    )

    stats = (
        dff_radar.groupby(["EloBin", "CrowdStatus"])["FTResult"]
        .apply(lambda x: (x == "A").mean() * 100)
        .unstack()
    )

    for col in ["Crowd (Full)", "Ghost (Empty)"]:
        if col not in stats.columns:
            stats[col] = 0

    if stats.empty:
        st.warning("No data available for the selected filters.")
    else:
        categories = stats.index.astype(str).tolist()
        values_crowd = stats["Crowd (Full)"].fillna(0).tolist()
        values_ghost = stats["Ghost (Empty)"].fillna(0).tolist()

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values_crowd,
                theta=categories,
                fill="toself",
                name="Crowd (Full)",
                line=dict(color="#2ecc71", width=2),
                fillcolor="rgba(46, 204, 113, 0.25)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values_ghost,
                theta=categories,
                fill="toself",
                name="Ghost (Empty)",
                line=dict(color="#e74c3c", width=2),
                fillcolor="rgba(231, 76, 60, 0.25)",
            )
        )

        fig_radar.update_layout(
            template="plotly_dark",
            title=dict(text="Upset Rate by Elo Strength and Crowd Presence", x=0.02, y=0.98),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
            margin=dict(l=30, r=30, t=60, b=70),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    range=[0, 25],
                    tickvals=[5, 10, 15, 20],
                    ticktext=["5%", "10%", "15%", "20%"],
                    gridcolor="rgba(255,255,255,0.12)",
                ),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            ),
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        **Interpretation**  
        - Each axis represents favorite strength (EloBin).  
        - Radius shows upset rate (home favorite loses).  
        - The red vs green areas highlight the crowd absence effect.  
        """)

# =========================================================
# PAGE: DEEP DIVE
# =========================================================
elif page == "🔍 Deep Dive":
    st.title("🔍 Team-Level Deep Dive Analysis")
    st.caption("Detailed analysis of crowd effect on individual teams, controlling for Elo strength.")
    
    st.info("""
    **What you're looking at:**
    - Each point = one team's average performance in Ghost vs Crowd conditions
    - X-axis = Elo difference between crowd and ghost conditions (how much stronger the home team faced in crowds)
    - Y-axis = **Residual** = actual crowd effect after controlling for opponent strength
    - Point size = total games analyzed for that team
    """)
    
    st.divider()
    
    # =========================================================
    # Sidebar filters (for Deep Dive page)
    # -------------------------
    st.sidebar.header("Deep Dive Filters")

    leagues_sel = st.sidebar.multiselect("Leagues", TOP_LEAGUES, default=TOP_LEAGUES)

    min_season = int(df["Season"].min())
    max_season = int(df["Season"].max())
    season_range = st.sidebar.slider("Season range", min_season, max_season, (min_season, max_season))

    df_f = df[df["Division"].isin(leagues_sel)].copy()
    df_f = df_f[(df_f["Season"] >= season_range[0]) & (df_f["Season"] <= season_range[1])].copy()

    st.sidebar.divider()
    st.sidebar.subheader("Crowd classification")
    fixed_low = st.sidebar.slider("Ghost threshold (AttendanceRate)", 0.00, 0.60, 0.30, 0.01)
    fixed_high = st.sidebar.slider("Crowd threshold (AttendanceRate)", 0.40, 1.00, 0.70, 0.01)

    min_matches_status = st.sidebar.slider("Min matches per status", 3, 30, 8, 1)

    # -------------------------
    # Define helper function with access to page variables
    # -------------------------
    def add_crowd_status(df_in: pd.DataFrame) -> pd.DataFrame:
        d = df_in.copy()

        d["thr_low"] = fixed_low
        d["thr_high"] = fixed_high
        conds = [
            d["AttendanceRate"] <= fixed_low,
            d["AttendanceRate"] >= fixed_high,
        ]

        d["CrowdStatus"] = np.select(conds, ["Ghost", "Crowd"], default="Partial").astype("object")
        return d

    # =========================================================
    # A1: Residual scatter (control for Elo)
    # =========================================================
    st.header("Team-Level Analysis: Residual Crowd Effect (controlling for Elo)")

    d1 = add_crowd_status(df_f)
    d1 = d1[d1["CrowdStatus"].isin(["Ghost", "Crowd"])].copy()

    # aggregate per team×status (HOME only)
    team_stats = (
        d1.groupby(["Division", "HomeTeam", "CrowdStatus"])
          .agg(
              WinPct=("HomeWin", "mean"),
              AvgHomeElo=("HomeElo", "mean"),
              Games=("HomeWin", "size")
          )
          .reset_index()
    )
    team_stats["WinPct"] *= 100

    wide = team_stats.pivot_table(
        index=["Division", "HomeTeam"],
        columns="CrowdStatus",
        values=["WinPct", "AvgHomeElo", "Games"]
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index(drop=False)  # Keep Division and HomeTeam as columns
    wide = wide.reset_index(drop=True)   # Reset the default index

    # require enough games in both statuses
    wide = wide[
        (wide["Games_Crowd"] >= min_matches_status) &
        (wide["Games_Ghost"] >= min_matches_status)
    ].copy()

    # Ensure no duplicate rows
    wide = wide.drop_duplicates(subset=["Division", "HomeTeam"]).reset_index(drop=True)

    if len(wide) < 5:
        st.warning("Not enough teams after filters. Try lowering min matches or expanding season/leagues.")
    else:
        wide["DeltaWin_pp"] = wide["WinPct_Crowd"] - wide["WinPct_Ghost"]
        wide["DeltaElo"] = wide["AvgHomeElo_Crowd"] - wide["AvgHomeElo_Ghost"]
        wide["TotalGames"] = wide["Games_Crowd"] + wide["Games_Ghost"]

        # regression: DeltaWin ~ DeltaElo
        X = wide[["DeltaElo"]].values
        y = wide["DeltaWin_pp"].values
        model = LinearRegression()
        model.fit(X, y)
        wide["Expected_DeltaWin"] = model.predict(X)
        wide["Residual"] = wide["DeltaWin_pp"] - wide["Expected_DeltaWin"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Teams included", f"{len(wide)}")
        c2.metric("Slope (pp per Elo)", f"{model.coef_[0]:.3f}")
        c3.metric("Intercept", f"{model.intercept_:.3f}")

        # team selector
        wide["TeamLabel"] = wide["HomeTeam"] + " (" + wide["Division"] + ")"
        
        # Get unique team labels and sort them for display
        team_labels_sorted = sorted(wide["TeamLabel"].unique().tolist())
        selected = st.multiselect("Highlight teams", team_labels_sorted, default=[])

        # Create Selected column BEFORE creating the figure
        wide["Selected"] = wide["TeamLabel"].isin(selected)

        fig = px.scatter(
            wide,
            x="DeltaElo",
            y="Residual",
            color="Division",
            size="TotalGames",
            hover_name="HomeTeam",
            custom_data=["Selected", "TeamLabel"],
            hover_data={
                "Division": True,
                "DeltaElo": ":.1f",
                "DeltaWin_pp": ":.1f",
                "Expected_DeltaWin": ":.1f",
                "Residual": ":.1f",
                "WinPct_Crowd": ":.1f",
                "WinPct_Ghost": ":.1f",
                "AvgHomeElo_Crowd": ":.1f",
                "AvgHomeElo_Ghost": ":.1f",
                "Games_Crowd": True,
                "Games_Ghost": True,
                "Selected": False,
                "TeamLabel": False,
            },
            title="Residual Crowd Effect vs ΔElo (teams)"
        )
        fig.add_hline(y=0)
        fig.add_vline(x=0)

        # highlight selected - dim non-selected and highlight selected
        if len(selected) > 0:
            # Dim all traces
            fig.update_traces(marker=dict(opacity=0.2), selector=dict(mode="markers"))
            
            # Add bright overlay for selected points
            selected_points = wide[wide["Selected"]].copy()
            
            fig_selected = px.scatter(
                selected_points,
                x="DeltaElo",
                y="Residual",
                color="Division",
                size="TotalGames",
                hover_name="HomeTeam",
            )
            
            # Add all traces from the selected figure (handles multiple divisions)
            for trace in fig_selected.data:
                trace.marker.opacity = 1.0
                trace.showlegend = False
                fig.add_trace(trace)

        st.plotly_chart(fig, width="stretch")

        with st.expander("Underlying table"):
            cols = ["Division","HomeTeam","Residual","DeltaWin_pp","DeltaElo","Games_Crowd","Games_Ghost","WinPct_Crowd","WinPct_Ghost"]
            st.dataframe(wide[cols].sort_values("Residual", ascending=False), width="stretch")

        st.divider()

        # =========================================================
        # A2: PPG Ghost vs Crowd (match screenshot)
        # =========================================================
        st.header("Team Performance: Ghost vs Crowd (Points Per Game)")
        st.caption("Scatter plot comparing each team's average points per game in Ghost vs Crowd conditions. Points above the diagonal indicate a crowd boost.")

        dppg = d1.copy()
        dppg["HomePoints"] = np.select(
            [dppg["FTResult"] == "H", dppg["FTResult"] == "D"],
            [3, 1],
            default=0
        )

        team_ppg = (
            dppg.groupby(["Division", "HomeTeam", "CrowdStatus"], as_index=False)
                .agg(PPG=("HomePoints", "mean"), Games=("HomePoints", "size"))
        )

        wide_ppg = team_ppg.pivot_table(
            index=["Division", "HomeTeam"],
            columns="CrowdStatus",
            values=["PPG", "Games"]
        )
        wide_ppg.columns = [f"{a}_{b}" for a, b in wide_ppg.columns]
        wide_ppg = wide_ppg.reset_index(drop=False)

        required_cols = {"PPG_Crowd", "PPG_Ghost", "Games_Crowd", "Games_Ghost"}
        if not required_cols.issubset(set(wide_ppg.columns)):
            st.warning("Not enough data to build Ghost vs Crowd PPG comparison after filtering.")
        else:
            wide_ppg = wide_ppg[
                (wide_ppg["Games_Crowd"] >= min_matches_status) &
                (wide_ppg["Games_Ghost"] >= min_matches_status)
            ].copy()

            if wide_ppg.empty:
                st.warning("Not enough teams after filters. Try lowering min matches or expanding season/leagues.")
            else:
                max_ppg = float(max(wide_ppg["PPG_Crowd"].max(), wide_ppg["PPG_Ghost"].max(), 3.0))

                fig_ppg = px.scatter(
                    wide_ppg,
                    x="PPG_Crowd",
                    y="PPG_Ghost",
                    color="Division",
                    hover_name="HomeTeam",
                    hover_data={
                        "Division": True,
                        "PPG_Crowd": ":.2f",
                        "PPG_Ghost": ":.2f",
                        "Games_Crowd": True,
                        "Games_Ghost": True,
                    },
                    title="Team PPG: Crowd vs Ghost Attendance",
                )

                fig_ppg.add_trace(
                    go.Scatter(
                        x=[0, max_ppg],
                        y=[0, max_ppg],
                        mode="lines",
                        line=dict(color="rgba(200,200,200,0.45)", dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig_ppg.update_layout(
                    template="plotly_dark",
                    height=520,
                    margin=dict(l=60, r=40, t=80, b=60),
                    legend_title_text="Division",
                    xaxis=dict(
                        title="PPG (Crowd)",
                        range=[0, max_ppg],
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.12)",
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title="PPG (Ghost)",
                        range=[0, max_ppg],
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.12)",
                        zeroline=False,
                    ),
                )

                st.plotly_chart(fig_ppg, use_container_width=True)

# =========================================================
# PAGE: ELO VS ATTENDANCE (Gapminder)
# =========================================================
elif page == "⚽ Elo vs Attendance":
    st.title("Elo vs Attendance (Gapminder-style)")

    # -------------------------
    # Config
    # -------------------------
    DEFAULT_CSV_ELO = DATA_PATH
    TOP_LEAGUES_ELO = TOP_LEAGUES
    SEASON_MIN, SEASON_MAX = 2016, 2024

    # -------------------------
    # Data prep
    # -------------------------
    @st.cache_data
    def build_team_season(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["MatchDate"])

        # coerce numeric
        for c in ["Attendance", "HomeElo", "AwayElo"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Season (football season starts ~Aug) - only if not provided
        if "Season" not in df.columns:
            y = df["MatchDate"].dt.year
            m = df["MatchDate"].dt.month
            df["Season"] = np.where(m >= 8, y, y - 1).astype(int)

        # keep required columns (need FTResult to compute points)
        needed = ["Division", "Season", "HomeTeam", "AwayTeam", "HomeElo", "AwayElo", "Attendance", "FTResult"]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        df = df[needed].copy()

        # compute points per match
        df["Points_Home"] = np.where(df["FTResult"] == "H", 3, np.where(df["FTResult"] == "D", 1, 0))
        df["Points_Away"] = np.where(df["FTResult"] == "A", 3, np.where(df["FTResult"] == "D", 1, 0))

        # long format: home + away -> unified Team/Elo/Points
        home = df.rename(columns={"HomeTeam": "Team", "HomeElo": "Elo", "Points_Home": "Points"})[
            ["Division", "Season", "Team", "Elo", "Attendance", "Points"]
        ]
        away = df.rename(columns={"AwayTeam": "Team", "AwayElo": "Elo", "Points_Away": "Points"})[
            ["Division", "Season", "Team", "Elo", "Attendance", "Points"]
        ]
        long_df = pd.concat([home, away], ignore_index=True)

        # minimal cleaning: keep matches even if attendance is missing/zero (e.g., 2020 closed-door games)
        long_df = long_df.dropna(subset=["Division", "Season", "Team", "Elo", "Points"])

        # aggregate: Team x Season x League (per season points sum)
        ts = (
            long_df.groupby(["Division", "Season", "Team"], as_index=False)
            .agg(
                Elo=("Elo", "mean"),
                Attendance=("Attendance", "mean"),
                Points=("Points", "sum"),
                Matches=("Elo", "size"),
            )
        )
        return ts

    def year_watermark_annotation(year_value) -> dict:
        return dict(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=str(year_value),
            showarrow=False,
            font=dict(size=140, color="rgba(120,120,120,0.20)"),
            align="center"
        )

    def build_gapminder_like_figure(
        ts: pd.DataFrame,
        leagues_sel: list,
        highlight_teams: list
    ) -> go.Figure:
        dff = ts[ts["Division"].isin(leagues_sel)].copy()
        seasons = sorted(dff["Season"].unique().tolist())
        if not seasons:
            return go.Figure()

        # axis ranges (stable across time) - align to (0,0) with a little zoom out
        x_min = 0.0
        x_max_raw = float(dff["Points"].max())
        y_min = 0.0
        y_max_raw = float(dff["Attendance"].max())

        x_max = x_max_raw * 1.08
        y_max = y_max_raw * 1.08

        # marker size scaling based on Elo range
        elo_min = float(dff["Elo"].min())
        elo_max = float(dff["Elo"].max())

        def scale_size(series: pd.Series) -> np.ndarray:
            # map Elo to a visible size range, fallback if all equal
            if elo_max - elo_min <= 1e-9:
                return np.full(len(series), 26.0)
            return np.interp(series, [elo_min, elo_max], [14.0, 60.0])

        # color map for leagues
        palette = px.colors.qualitative.Plotly
        leagues_sorted = sorted(leagues_sel)
        color_map = {lg: palette[i % len(palette)] for i, lg in enumerate(leagues_sorted)}

        has_focus = (len(highlight_teams) > 0)

        def traces_for_season(season_value: int, show_legend: bool) -> list:
            ds = dff[dff["Season"] == season_value].copy()

            # NORMAL MODE: if no highlighted teams, do NOT fade anything
            if not has_focus:
                out = []
                for lg in leagues_sorted:
                    part = ds[ds["Division"] == lg]
                    out.append(
                        go.Scatter(
                            x=part["Points"],
                            y=part["Attendance"],
                            mode="markers",
                            marker=dict(
                                size=scale_size(part["Elo"]).tolist(),
                                sizemode="diameter",
                                color=color_map[lg],
                                opacity=0.9,
                                line=dict(width=0),
                            ),
                            name=lg,
                            legendgroup=lg,
                            showlegend=show_legend,
                            customdata=np.stack([part["Team"], part["Matches"], part["Elo"], part["Points"]], axis=1) if len(part) else None,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "League: " + lg + "<br>"
                                "Points: %{x:.0f}<br>"
                                "Attendance: %{y:.0f}<br>"
                                "Elo (bubble): %{customdata[2]:.1f}<br>"
                                "Matches: %{customdata[1]}<extra></extra>"
                            ),
                        )
                    )
                return out

            # FOCUS MODE: fade non-highlight + highlight selected
            ds["Highlighted"] = ds["Team"].isin(highlight_teams)

            out = []
            for lg in leagues_sorted:
                part = ds[ds["Division"] == lg]
                non = part[~part["Highlighted"]]
                hi = part[part["Highlighted"]]

                out.append(
                    go.Scatter(
                        x=non["Points"],
                        y=non["Attendance"],
                        mode="markers",
                        marker=dict(
                            size=scale_size(non["Elo"]).tolist(),
                            sizemode="diameter",
                            color=color_map[lg],
                            opacity=0.15,
                            line=dict(width=0),
                        ),
                        name=lg,
                        legendgroup=lg,
                        showlegend=show_legend,
                        customdata=np.stack([non["Team"], non["Matches"], non["Elo"], non["Points"]], axis=1) if len(non) else None,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "League: " + lg + "<br>"
                            "Points: %{x:.0f}<br>"
                            "Attendance: %{y:.0f}<br>"
                            "Elo (bubble): %{customdata[2]:.1f}<br>"
                            "Matches: %{customdata[1]}<extra></extra>"
                        ),
                    )
                )

                out.append(
                    go.Scatter(
                        x=hi["Points"],
                        y=hi["Attendance"],
                        mode="markers",
                        marker=dict(
                            size=scale_size(hi["Elo"]).tolist(),
                            sizemode="diameter",
                            color=color_map[lg],
                            opacity=1.0,
                            line=dict(width=2, color="black"),
                        ),
                        name=lg,
                        legendgroup=lg,
                        showlegend=False,
                        customdata=np.stack([hi["Team"], hi["Matches"], hi["Elo"], hi["Points"]], axis=1) if len(hi) else None,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "League: " + lg + "<br>"
                            "Points: %{x:.0f}<br>"
                            "Attendance: %{y:.0f}<br>"
                            "Elo (bubble): %{customdata[2]:.1f}<br>"
                            "Matches: %{customdata[1]}<extra></extra>"
                        ),
                    )
                )
            return out

        # Base (first season)
        base_season = seasons[0]
        fig = go.Figure(data=traces_for_season(base_season, show_legend=True))

        # Frames (one per season)
        frames = []
        for s in seasons:
            frames.append(
                go.Frame(
                    name=str(s),
                    data=traces_for_season(s, show_legend=True),
                    layout=go.Layout(annotations=[year_watermark_annotation(s)])
                )
            )
        fig.frames = frames

        # Slider steps
        steps = []
        for s in seasons:
            steps.append(
                dict(
                    method="animate",
                    args=[[str(s)], {"mode": "immediate",
                                    "frame": {"duration": 600, "redraw": True},
                                    "transition": {"duration": 200}}],
                    label=str(s)
                )
            )

        # Layout like Gapminder
        fig.update_layout(
            height=720,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=40, b=60),
            legend_title_text="League",
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(color="black")
            ),
            font=dict(color="black"),
            xaxis=dict(
                title=dict(text="Points per season", font=dict(color="black")),
                range=[x_min, x_max],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
                tickfont=dict(color="black"),
            ),
            yaxis=dict(
                title=dict(text="Attendance (avg per match)", font=dict(color="black")),
                range=[y_min, y_max],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
                tickfont=dict(color="black"),
            ),
            annotations=[year_watermark_annotation(base_season)],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.0,
                    y=-0.12,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 600, "redraw": True},
                                    "transition": {"duration": 200},
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.12,
                    y=-0.12,
                    len=0.85,
                    currentvalue=dict(prefix="Season: ", visible=True),
                    steps=steps,
                )
            ],
        )

        return fig

    # -------------------------
    # App
    # -------------------------
    ts = build_team_season(DEFAULT_CSV_ELO)

    # Seasons 2005-2024 only
    ts = ts[(ts["Season"] >= SEASON_MIN) & (ts["Season"] <= SEASON_MAX)].copy()

    # ---- Leagues filter ----
    all_leagues = sorted(ts["Division"].unique().tolist())
    default_leagues = [lg for lg in TOP_LEAGUES_ELO if lg in all_leagues] or all_leagues

    st.sidebar.markdown("### Leagues (Division)")
    leagues_sel = st.sidebar.multiselect(" ", all_leagues, default=default_leagues)

    if not leagues_sel:
        st.warning("בחר לפחות ליגה אחת.")
        st.stop()

    # ---- Teams list (default: all checked; highlight only when NOT all checked) ----
    ts_leagues = ts[ts["Division"].isin(leagues_sel)].copy()
    teams_all = sorted(ts_leagues["Team"].unique().tolist())

    st.sidebar.markdown("### Teams (✓)")
    st.sidebar.caption("ברירת מחדל: כולם מסומנים. כדי להאיר קבוצה, הורד ✓ מכל היתר והשאר רק את מי שרוצים לבדוק.")

    # ---- Quick actions ----
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Clear all ✓"):
            st.session_state["teams_pick_df"] = pd.DataFrame(
                {"Show": [False] * len(teams_all), "Team": teams_all}
            )

    with col2:
        if st.button("Select all ✓"):
            st.session_state["teams_pick_df"] = pd.DataFrame(
                {"Show": [True] * len(teams_all), "Team": teams_all}
            )

    q = st.sidebar.text_input("Search team")

    # session state: default True for all teams
    if "teams_pick_df" not in st.session_state:
        st.session_state["teams_pick_df"] = pd.DataFrame(
            {"Show": [True] * len(teams_all), "Team": teams_all}
        )
    else:
        old = st.session_state["teams_pick_df"]
        old_map = dict(zip(old["Team"], old["Show"]))
        st.session_state["teams_pick_df"] = pd.DataFrame(
            {"Show": [old_map.get(t, True) for t in teams_all], "Team": teams_all}
        )

    view_df = st.session_state["teams_pick_df"]
    if q:
        view_df = view_df[view_df["Team"].str.contains(q, case=False, na=False)].copy()

    edited = st.sidebar.data_editor(
        view_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        height=420,
        column_config={
            "Show": st.column_config.CheckboxColumn("✓", help="Selected"),
            "Team": st.column_config.TextColumn("Team", disabled=True),
        },
        disabled=["Team"],
    )

    # write back edits
    full_df = st.session_state["teams_pick_df"].copy()
    edited_map = dict(zip(edited["Team"], edited["Show"]))
    full_df["Show"] = full_df.apply(lambda r: edited_map.get(r["Team"], r["Show"]), axis=1)
    st.session_state["teams_pick_df"] = full_df

    selected = full_df.loc[full_df["Show"], "Team"].tolist()

    # KEY RULE:
    # - default (all checked) => highlight_teams = []
    # - only if user narrowed down (not all, not empty) => highlight these teams
    highlight_teams = []
    if 0 < len(selected) < len(teams_all):
        highlight_teams = selected

    fig = build_gapminder_like_figure(ts, leagues_sel, highlight_teams)
    st.plotly_chart(fig, use_container_width=True)
