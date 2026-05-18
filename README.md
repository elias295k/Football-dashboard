# Football Crowd Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-dashboard-pexu4geamjr.streamlit.app/)

An interactive Streamlit dashboard for analyzing how football crowd attendance relates to home advantage, team performance, and Elo-based expectations across major European leagues.

The project explores whether high-attendance matches are associated with stronger home performance, while using Elo ratings to reduce confusion between crowd effects and team-strength differences.

---

## Live Demo

The dashboard is available online here:

[https://football-dashboard-pexu4geamjr.streamlit.app/](https://football-dashboard-pexu4geamjr.streamlit.app/)

---

## Project Overview

This dashboard analyzes historical football match data and focuses on one main question:

> Does crowd attendance strengthen home advantage in football?

The app compares matches under low-crowd and high-crowd conditions, evaluates home win percentage, analyzes upset rates, and provides team-level views that account for Elo rating differences.

The dashboard is designed for exploratory sports analytics and includes multiple interactive pages, filters, and Plotly visualizations.

---

## Main Features

### Home Page

* Summary metrics for total matches, analyzed leagues, and covered seasons.
* Quick comparison between low-crowd and high-crowd home win percentages.
* Explanation of the project question, key concepts, and interpretation guidelines.

### High-Level Analysis

* League-level comparison of home win percentage under low-crowd vs high-crowd conditions.
* Interactive crowd-threshold sliders.
* Dumbbell chart showing crowd effect by league.
* Radar chart showing upset rates by favorite strength using Elo differences.

### Team-Level Deep Dive

* Team-level analysis of crowd impact beyond Elo-based team strength.
* Residual analysis using linear regression to estimate performance changes not explained by Elo changes.
* Scatter plot for identifying teams that overperform or underperform under high-crowd conditions.
* Team points-per-game comparison between high-crowd and low-crowd matches.
* Filters for leagues, seasons, and minimum match counts.

### Elo vs Attendance

* Gapminder-style animated bubble chart showing how attendance and performance evolve over time.
* Season-by-season animation from 2016 to 2024.
* Team selection and highlighting.
* Bubble size based on average Elo rating.

---

## Tech Stack

* **Python**
* **Streamlit** — interactive dashboard framework
* **Pandas** — data loading, cleaning, and aggregation
* **NumPy** — numerical operations
* **Plotly** — interactive visualizations
* **Scikit-learn** — linear regression model for residual analysis

---

## Project Structure

```text
Football-dashboard/
├── streamlit_app.py
├── requirements.txt
├── data/
│   └── Matches_clean_final.csv
└── README.md
```

---

## Dataset

The dashboard uses a cleaned match dataset located at:

```text
data/Matches_clean_final.csv
```

The app expects the dataset to include match-level information such as:

* Match date
* League/division
* Season
* Home and away teams
* Full-time result
* Attendance
* Stadium capacity
* Home and away Elo ratings

The dashboard focuses on selected top European league codes:

```python
["E0", "SP1", "I1", "D1", "F1"]
```

---

## Key Metrics

### Attendance Rate

```text
Attendance Rate = Attendance / Stadium Capacity
```

Used to classify matches into low-crowd and high-crowd conditions.

### Home Win Percentage

```text
Home Win % = percentage of matches won by the home team
```

Used as a main indicator of home advantage.

### Points Per Game

```text
Win = 3 points
Draw = 1 point
Loss = 0 points
```

Used to compare team performance under different crowd conditions.

### Elo Difference

```text
Elo Difference = Home Elo - Away Elo
```

Used to estimate the expected strength gap between teams.

### Residual Crowd Effect

The dashboard estimates whether a team performs better or worse in high-crowd matches after accounting for Elo changes.

This is done by fitting a linear regression model between Elo difference and win-percentage difference, then analyzing the residuals.

---

## Run Locally

The dashboard is already available online through the live demo link above.
Use the following steps only if you want to run the project locally or inspect the code.

Clone the repository:

```bash
git clone https://github.com/elias295k/Football-dashboard.git
cd Football-dashboard
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## How to Use the Dashboard

1. Start from the **Home** page to understand the main question and key concepts.
2. Open **Analysis** to compare low-crowd and high-crowd effects across leagues.
3. Use the threshold sliders to change the definitions of low and high attendance.
4. Open **Deep Dive** to inspect specific teams and control for Elo-based strength changes.
5. Open **Elo vs Attendance** to explore animated season-by-season changes in performance and attendance.

---

## Interpretation Notes

The dashboard shows associations in historical football data.
It does not prove that attendance directly causes better home performance.

To make the analysis more reliable, the dashboard uses Elo ratings and team-level comparisons to reduce the effect of team-strength differences.

---

## Future Improvements

* Add screenshots or GIFs of the main dashboard pages.
* Add league-name mapping instead of showing only division codes.
* Add more statistical tests for crowd-effect significance.
* Add a model comparison section for predicting match outcomes.
* Add support for uploading a custom dataset.

---

## Author

**Elias Kanbora**
Data Engineering Student
Ben-Gurion University of the Negev

GitHub: [@elias295k](https://github.com/elias295k)
