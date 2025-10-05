import numpy as np
import pandas as pd


def compute_weighted_lastN_form(df, N=3, decay=0.7):
    df = df.sort_values('Date')
    df['HomeResult'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
    df['AwayResult'] = df['FTR'].map({'H': -1, 'D': 0, 'A': 1})

    df['Home_LastNForm'] = 0.0
    df['Away_LastNForm'] = 0.0

    for team in df['HomeTeam'].unique():
        team_mask = df['HomeTeam'] == team
        results = df.loc[team_mask, 'HomeResult'].shift(1)
        weighted = results.rolling(N, min_periods=1).apply(
            lambda x: np.sum(x * decay ** np.arange(len(x) - 1, -1, -1)) / np.sum(decay ** np.arange(len(x))), raw=True
        )
        df.loc[team_mask, 'Home_LastNForm'] = weighted

    for team in df['AwayTeam'].unique():
        team_mask = df['AwayTeam'] == team
        results = df.loc[team_mask, 'AwayResult'].shift(1)
        weighted = results.rolling(N, min_periods=1).apply(
            lambda x: np.sum(x * decay ** np.arange(len(x) - 1, -1, -1)) / np.sum(decay ** np.arange(len(x))), raw=True
        )
        df.loc[team_mask, 'Away_LastNForm'] = weighted

    df.drop(columns=['HomeResult', 'AwayResult'], inplace=True)
    return df

def compute_avg_goals_N(df, n_matches=5):
    df = df.sort_values('Date')

    df[f'HomeTeam_AvgGoalsScored{n_matches}'] = 0.0
    df[f'HomeTeam_AvgGoalsConceded{n_matches}'] = 0.0
    df[f'AwayTeam_AvgGoalsScored{n_matches}'] = 0.0
    df[f'AwayTeam_AvgGoalsConceded{n_matches}'] = 0.0

    for team in df['HomeTeam'].unique():
        team_mask = df['HomeTeam'] == team
        df.loc[team_mask, f'HomeTeam_AvgGoalsScored{n_matches}'] = (
            df.loc[team_mask, 'FTHG'].shift(1).rolling(n_matches, min_periods=1).mean()
        )
        df.loc[team_mask, f'HomeTeam_AvgGoalsConceded{n_matches}'] = (
            df.loc[team_mask, 'FTAG'].shift(1).rolling(n_matches, min_periods=1).mean()
        )

    for team in df['AwayTeam'].unique():
        team_mask = df['AwayTeam'] == team
        df.loc[team_mask, f'AwayTeam_AvgGoalsScored{n_matches}'] = (
            df.loc[team_mask, 'FTAG'].shift(1).rolling(n_matches, min_periods=1).mean()
        )
        df.loc[team_mask, f'AwayTeam_AvgGoalsConceded{n_matches}'] = (
            df.loc[team_mask, 'FTHG'].shift(1).rolling(n_matches, min_periods=1).mean()
        )

    return df


def compute_head_to_head(df, last_n=3):

    df = df.sort_values('Date').copy()


    df['Home_H2H'] = 0.0
    df['Away_H2H'] = 0.0


    df['HomeResult'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
    df['AwayResult'] = df['FTR'].map({'H': -1, 'D': 0, 'A': 1})


    teams = df['HomeTeam'].unique()

    for home_team in teams:
        for away_team in teams:
            if home_team == away_team:
                continue
            mask = ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | \
                   ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
            h2h_matches = df.loc[mask].copy()

            h2h_matches['Home_H2H_temp'] = h2h_matches['HomeResult'].shift(1).rolling(last_n, min_periods=1).mean()
            h2h_matches['Away_H2H_temp'] = h2h_matches['AwayResult'].shift(1).rolling(last_n, min_periods=1).mean()

            df.loc[h2h_matches.index, 'Home_H2H'] = h2h_matches['Home_H2H_temp']
            df.loc[h2h_matches.index, 'Away_H2H'] = h2h_matches['Away_H2H_temp']

    df.drop(columns=['HomeResult', 'AwayResult'], inplace=True)

    return df

def add_team_features(df: pd.DataFrame, n_matches: int = 3) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = [
        'Home_ShotsPerMatch', 'Away_ShotsPerMatch',
        'Home_ShotsConcededPerMatch', 'Away_ShotsConcededPerMatch',
        'Home_CardsPerMatch', 'Away_CardsPerMatch',
        'Home_WinRate_Home', 'Away_WinRate_Away'
    ]
    for col in feature_cols:
        df[col] = np.nan

    teams = sorted(df['HomeTeam'].unique())

    for team in teams:

        home_matches = df[df['HomeTeam'] == team].sort_values('Date')

        df.loc[home_matches.index, 'Home_ShotsPerMatch'] = (
            home_matches['HS'].rolling(n_matches).mean()
        )
        df.loc[home_matches.index, 'Home_ShotsConcededPerMatch'] = (
            home_matches['AS'].rolling(n_matches).mean()
        )

        df.loc[home_matches.index, 'Home_CardsPerMatch'] = (
            (home_matches['HY'] + 2 * home_matches['HR']).rolling(n_matches).mean()
        )

        df.loc[home_matches.index, 'Home_WinRate_Home'] = (
            home_matches['FTR'].eq('H').rolling(n_matches).mean()
        )


        away_matches = df[df['AwayTeam'] == team].sort_values('Date')

        df.loc[away_matches.index, 'Away_ShotsPerMatch'] = (
            away_matches['AS'].rolling(n_matches).mean()
        )
        df.loc[away_matches.index, 'Away_ShotsConcededPerMatch'] = (
            away_matches['HS'].rolling(n_matches).mean()
        )
        df.loc[away_matches.index, 'Away_CardsPerMatch'] = (
            (away_matches['AY'] + 2 * away_matches['AR']).rolling(n_matches).mean()
        )

        df.loc[away_matches.index, 'Away_WinRate_Away'] = (
            away_matches['FTR'].eq('A').rolling(n_matches).mean()
        )

    for col in ['AvgH', 'AvgD', 'AvgA']:
        if col in df.columns:
            df[f'Prob_{col[-1]}'] = 1 / df[col]

    if all(f'Prob_{x}' in df.columns for x in ['H', 'D', 'A']):
        total = df['Prob_H'] + df['Prob_D'] + df['Prob_A']
        df['Prob_H'] /= total
        df['Prob_D'] /= total
        df['Prob_A'] /= total

    df.fillna(0, inplace=True)

    return df


if __name__ == "__main__":
    path = "../../data/processed/LaLiga_clean.csv"
    df = pd.read_csv(path, parse_dates=['Date'])
    df = add_team_features(df)
    print(df.head(10)[[
        'Date','HomeTeam','AwayTeam',
        'Home_ShotsPerMatch','Away_ShotsPerMatch',
        'Home_CardsPerMatch','Away_CardsPerMatch',
        'Home_WinRate_Home','Away_WinRate_Away'
    ]])
