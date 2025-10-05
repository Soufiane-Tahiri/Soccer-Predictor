import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime


class DynamicMatchPredictor:
    """Dynamic soccer match predictor with team selection menus."""

    def __init__(self, models_dir='../models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.df = None
        self.manual_data = []
        self.teams = []

    def load_models(self):
        """Load trained models."""
        model_files = {
            'RandomForest': 'RandomForest_LaLiga.pkl',
            'GradientBoosting': 'GradientBoosting_LaLiga.pkl',
            'LightGBM': 'LightGBM_LaLiga.pkl'
        }

        print("Loading models...")
        for name, fname in model_files.items():
            path = self.models_dir / fname
            if path.exists():
                self.models[name] = joblib.load(path)
                print(f"✓ {name}")

        if not self.models:
            raise FileNotFoundError("No models found!")
        print()

    def load_data(self, csv_path):
        """Load historical data and extract team list."""
        if csv_path and Path(csv_path).exists():
            print(f"Loading data from {csv_path}...")
            self.df = pd.read_csv(csv_path, parse_dates=['Date'])
            self.df = self.df.sort_values('Date').reset_index(drop=True)

            # Extract unique teams
            self.teams = sorted(set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique()))

            print(f"✓ Loaded {len(self.df)} historical matches")
            print(f"✓ Found {len(self.teams)} teams\n")
            return True
        else:
            print("⚠ No historical data file found")
            # Default La Liga teams
            self.teams = sorted([
                'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
                'Real Sociedad', 'Real Betis', 'Villarreal', 'Athletic Club',
                'Valencia', 'Osasuna', 'Celta Vigo', 'Rayo Vallecano',
                'Mallorca', 'Girona', 'Getafe', 'Alaves', 'Las Palmas',
                'Espanyol', 'Real Valladolid', 'Leganes'
            ])
            self.df = pd.DataFrame()
            print(f"✓ Using default team list ({len(self.teams)} teams)\n")
            return False

    def select_team(self, prompt, exclude=None):
        """Interactive team selection with menu."""
        available = [t for t in self.teams if t != exclude]

        print(f"\n{prompt}")
        print("─" * 60)

        # Display in columns for better readability
        for i in range(0, len(available), 2):
            left = f"{i + 1:2d}. {available[i]:<30}"
            right = f"{i + 2:2d}. {available[i + 1]}" if i + 1 < len(available) else ""
            print(f"{left} {right}")

        print("─" * 60)

        while True:
            try:
                choice = input(f"Select team (1-{len(available)}): ").strip()
                idx = int(choice) - 1

                if 0 <= idx < len(available):
                    selected = available[idx]
                    print(f"✓ Selected: {selected}")
                    return selected
                else:
                    print(f"❌ Please enter a number between 1 and {len(available)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                exit(0)

    def get_number_input(self, prompt, default, min_val=None, max_val=None):
        """Get validated number input."""
        while True:
            try:
                value = input(f"{prompt} (default {default}): ").strip()
                if not value:
                    return default

                num = float(value)

                if min_val is not None and num < min_val:
                    print(f"❌ Value must be at least {min_val}")
                    continue
                if max_val is not None and num > max_val:
                    print(f"❌ Value must be at most {max_val}")
                    continue

                return num
            except ValueError:
                print("❌ Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                exit(0)

    def add_manual_match(self):
        """Add a recent match with menu selection."""
        print("\n" + "=" * 60)
        print("ADD RECENT MATCH")
        print("=" * 60)

        home_team = self.select_team("🏠 SELECT HOME TEAM")
        away_team = self.select_team("✈️  SELECT AWAY TEAM", exclude=home_team)

        print(f"\nMatch: {home_team} vs {away_team}")
        print("─" * 60)

        h_goals = int(self.get_number_input("Home goals", 0, 0, 20))
        a_goals = int(self.get_number_input("Away goals", 0, 0, 20))

        add_details = input("\nAdd detailed stats (shots/cards)? (y/N): ").strip().lower()

        if add_details == 'y':
            h_shots = int(self.get_number_input("Home shots", 12, 0, 50))
            a_shots = int(self.get_number_input("Away shots", 10, 0, 50))
            h_cards = int(self.get_number_input("Home yellow cards", 2, 0, 10))
            a_cards = int(self.get_number_input("Away yellow cards", 2, 0, 10))
        else:
            h_shots = a_shots = h_cards = a_cards = None

        # Determine result
        if h_goals > a_goals:
            result = 'H'
        elif h_goals < a_goals:
            result = 'A'
        else:
            result = 'D'

        match_data = {
            'Date': pd.to_datetime(datetime.now()),
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': h_goals,
            'FTAG': a_goals,
            'FTR': result,
            'HS': h_shots if h_shots is not None else 12,
            'AS': a_shots if a_shots is not None else 10,
            'HY': h_cards if h_cards is not None else 2,
            'AY': a_cards if a_cards is not None else 2,
            'HR': 0,
            'AR': 0,
            'AvgH': 2.0,
            'AvgD': 3.5,
            'AvgA': 3.0
        }

        self.manual_data.append(match_data)
        print(f"\n✓ Added: {home_team} {h_goals}-{a_goals} {away_team}")

    def get_combined_data(self):
        """Combine historical and manual data."""
        if not self.manual_data:
            return self.df.copy() if self.df is not None else pd.DataFrame()

        manual_df = pd.DataFrame(self.manual_data)

        if self.df is not None and len(self.df) > 0:
            combined = pd.concat([self.df, manual_df], ignore_index=True)
        else:
            combined = manual_df

        return combined.sort_values('Date').reset_index(drop=True)

    def get_team_stats(self, team, is_home=True, n_matches=5):
        """Get team statistics from recent matches."""
        df = self.get_combined_data()

        if df is None or len(df) == 0:
            return {
                'form': 0.0,
                'goals_scored': 1.5,
                'goals_conceded': 1.5,
                'shots': 12 if is_home else 10,
                'shots_conceded': 10 if is_home else 12,
                'cards': 2.5,
                'win_rate': 0.33
            }

        if is_home:
            matches = df[df['HomeTeam'] == team].tail(n_matches)
            if len(matches) == 0:
                return self.get_team_stats(team, is_home, n_matches)

            wins = (matches['FTR'] == 'H').sum()
            losses = (matches['FTR'] == 'A').sum()

            return {
                'form': (wins - losses) / len(matches),
                'goals_scored': matches['FTHG'].mean(),
                'goals_conceded': matches['FTAG'].mean(),
                'shots': matches['HS'].mean(),
                'shots_conceded': matches['AS'].mean(),
                'cards': (matches['HY'].mean() + 2 * matches['HR'].mean()),
                'win_rate': wins / len(matches)
            }
        else:
            matches = df[df['AwayTeam'] == team].tail(n_matches)
            if len(matches) == 0:
                return self.get_team_stats(team, is_home, n_matches)

            wins = (matches['FTR'] == 'A').sum()
            losses = (matches['FTR'] == 'H').sum()

            return {
                'form': (wins - losses) / len(matches),
                'goals_scored': matches['FTAG'].mean(),
                'goals_conceded': matches['FTHG'].mean(),
                'shots': matches['AS'].mean(),
                'shots_conceded': matches['HS'].mean(),
                'cards': (matches['AY'].mean() + 2 * matches['AR'].mean()),
                'win_rate': wins / len(matches)
            }

    def get_h2h(self, home_team, away_team, n_matches=5):
        """Get head-to-head statistics."""
        df = self.get_combined_data()

        if df is None or len(df) == 0:
            return 0.0, 0.0

        h2h = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
            ].tail(n_matches)

        if len(h2h) == 0:
            return 0.0, 0.0

        home_wins = ((h2h['HomeTeam'] == home_team) & (h2h['FTR'] == 'H')).sum()
        away_wins = ((h2h['AwayTeam'] == home_team) & (h2h['FTR'] == 'A')).sum()

        home_score = (home_wins - away_wins) / len(h2h)
        away_score = -home_score

        return home_score, away_score

    def create_features(self, home_team, away_team, n_matches=5,
                        odds_home=2.0, odds_draw=3.5, odds_away=3.0):
        """Create feature dataframe for prediction."""

        print(f"\n{'=' * 60}")
        print(f"ANALYZING: {home_team} vs {away_team}")
        print(f"{'=' * 60}")
        print(f"Using last {n_matches} matches per team")

        if self.manual_data:
            print(f"Including {len(self.manual_data)} manually added match(es)")

        home_stats = self.get_team_stats(home_team, is_home=True, n_matches=n_matches)
        away_stats = self.get_team_stats(away_team, is_home=False, n_matches=n_matches)
        h2h_home, h2h_away = self.get_h2h(home_team, away_team, n_matches)

        # Convert odds to probabilities
        prob_h = 1 / odds_home
        prob_d = 1 / odds_draw
        prob_a = 1 / odds_away
        total = prob_h + prob_d + prob_a
        prob_h, prob_d, prob_a = prob_h / total, prob_d / total, prob_a / total

        print(f"\n{home_team} (Home):")
        print(
            f"  Form: {home_stats['form']:+.2f} | Goals: {home_stats['goals_scored']:.1f} | Win Rate: {home_stats['win_rate']:.1%}")
        print(f"\n{away_team} (Away):")
        print(
            f"  Form: {away_stats['form']:+.2f} | Goals: {away_stats['goals_scored']:.1f} | Win Rate: {away_stats['win_rate']:.1%}")
        print(f"\nH2H: {h2h_home:+.2f} (Home) vs {h2h_away:+.2f} (Away)")
        print(f"Odds: {odds_home:.2f} / {odds_draw:.2f} / {odds_away:.2f}")

        features = pd.DataFrame([{
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Home_LastNForm': home_stats['form'],
            'Away_LastNForm': away_stats['form'],
            'HomeTeam_AvgGoalsScored3': home_stats['goals_scored'],
            'AwayTeam_AvgGoalsScored3': away_stats['goals_scored'],
            'HomeTeam_AvgGoalsConceded3': home_stats['goals_conceded'],
            'AwayTeam_AvgGoalsConceded3': away_stats['goals_conceded'],
            'Home_H2H': h2h_home,
            'Away_H2H': h2h_away,
            'Home_ShotsPerMatch': home_stats['shots'],
            'Away_ShotsPerMatch': away_stats['shots'],
            'Home_ShotsConcededPerMatch': home_stats['shots_conceded'],
            'Away_ShotsConcededPerMatch': away_stats['shots_conceded'],
            'Home_CardsPerMatch': home_stats['cards'],
            'Away_CardsPerMatch': away_stats['cards'],
            'Home_WinRate_Home': home_stats['win_rate'],
            'Away_WinRate_Away': away_stats['win_rate'],
            'Prob_H': prob_h,
            'Prob_D': prob_d,
            'Prob_A': prob_a,
        }])

        return features

    def predict(self, home_team, away_team, n_matches=5,
                odds_home=2.0, odds_draw=3.5, odds_away=3.0):
        """Make prediction."""

        X = self.create_features(home_team, away_team, n_matches,
                                 odds_home, odds_draw, odds_away)

        print(f"\n{'=' * 60}")
        print("PREDICTIONS")
        print("=" * 60)

        results = {}
        outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}

        for name, model in self.models.items():
            # Check if this is LightGBM Booster (not sklearn wrapper)
            is_lgb_booster = 'Booster' in str(type(model))

            if is_lgb_booster:
                # LightGBM native model - needs one-hot encoding
                X_encoded = pd.get_dummies(X)

                # Get all possible team columns from training
                all_teams = self.teams
                for team in all_teams:
                    home_col = f'HomeTeam_{team}'
                    away_col = f'AwayTeam_{team}'
                    if home_col not in X_encoded.columns:
                        X_encoded[home_col] = 0
                    if away_col not in X_encoded.columns:
                        X_encoded[away_col] = 0

                # Set the correct team to 1
                X_encoded[f'HomeTeam_{home_team}'] = 1
                X_encoded[f'AwayTeam_{away_team}'] = 1

                # Sort columns to match training order
                X_aligned = X_encoded.reindex(sorted(X_encoded.columns), axis=1)

                # LightGBM Booster.predict returns probabilities directly
                proba = model.predict(X_aligned)[0]
                pred_idx = np.argmax(proba)

                # Map to class labels (A=0, D=1, H=2 based on label encoding)
                classes_map = {0: 'A', 1: 'D', 2: 'H'}
                pred = classes_map[pred_idx]
            else:
                # One-hot encoding for sklearn models
                X_encoded = pd.get_dummies(X)

                if hasattr(model, 'feature_names_in_'):
                    X_aligned = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
                else:
                    X_aligned = X_encoded

                pred = model.predict(X_aligned)[0]
                proba = model.predict_proba(X_aligned)[0]

            if hasattr(model, 'classes_'):
                classes = model.classes_
                pred_label = outcome_map.get(pred, pred)

                h_idx = list(classes).index('H')
                d_idx = list(classes).index('D')
                a_idx = list(classes).index('A')

                prob_home = proba[h_idx]
                prob_draw = proba[d_idx]
                prob_away = proba[a_idx]
            elif 'LightGBM' in name or 'Booster' in str(type(model)):
                # Already handled above
                pred_label = outcome_map.get(pred, pred)
                prob_home = proba[2]  # H
                prob_draw = proba[1]  # D
                prob_away = proba[0]  # A
            else:
                pred_label = outcome_map.get(pred, str(pred))
                prob_home, prob_draw, prob_away = proba[0], proba[1], proba[2]

            print(f"\n{name}:")
            print(f"  → {pred_label} ({max(proba) * 100:.1f}% confidence)")
            print(f"  H: {prob_home:.1%} | D: {prob_draw:.1%} | A: {prob_away:.1%}")

            results[name] = {
                'prediction': pred_label,
                'probabilities': {'H': prob_home, 'D': prob_draw, 'A': prob_away},
                'confidence': max(proba)
            }

        print(f"\n{'=' * 60}")
        print("ENSEMBLE PREDICTION")
        print("=" * 60)

        avg_h = np.mean([r['probabilities']['H'] for r in results.values()])
        avg_d = np.mean([r['probabilities']['D'] for r in results.values()])
        avg_a = np.mean([r['probabilities']['A'] for r in results.values()])

        ensemble_pred = outcome_map[max([('H', avg_h), ('D', avg_d), ('A', avg_a)], key=lambda x: x[1])[0]]
        max_prob = max(avg_h, avg_d, avg_a)

        print(f"\n  → {ensemble_pred} ({max_prob:.1%} confidence)")
        print(f"  H: {avg_h:.1%} | D: {avg_d:.1%} | A: {avg_a:.1%}")
        print("=" * 60 + "\n")

        return results


def main():
    """Interactive usage with team selection menus."""
    predictor = DynamicMatchPredictor(models_dir='../models')
    predictor.load_models()
    predictor.load_data('../data/processed/LaLiga_clean.csv')

    print("=" * 60)
    print("⚽ DYNAMIC MATCH PREDICTOR")
    print("=" * 60)
    print("\nAdd recent matches to improve predictions with latest form.\n")

    # Ask if user wants to add recent matches
    add_matches = input("Add recent match data? (y/N): ").strip().lower()

    if add_matches == 'y':
        while True:
            predictor.add_manual_match()

            another = input("\nAdd another match? (y/N): ").strip().lower()
            if another != 'y':
                break

    # Make prediction
    print(f"\n{'=' * 60}")
    print("MAKE PREDICTION")
    print("=" * 60)

    home_team = predictor.select_team("🏠 SELECT HOME TEAM")
    away_team = predictor.select_team("✈️  SELECT AWAY TEAM", exclude=home_team)

    print(f"\n{'─' * 60}")
    print("PREDICTION PARAMETERS")
    print("─" * 60)

    n = int(predictor.get_number_input("Recent matches to analyze", 5, 1, 20))
    odds_h = predictor.get_number_input("Home win odds", 2.0, 1.01, 100)
    odds_d = predictor.get_number_input("Draw odds", 3.5, 1.01, 100)
    odds_a = predictor.get_number_input("Away win odds", 3.0, 1.01, 100)

    predictor.predict(home_team, away_team, n, odds_h, odds_d, odds_a)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nGoodbye!")