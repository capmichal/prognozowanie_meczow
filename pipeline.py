import pandas as pd

class BettingEngine:
    def __init__(self, path_results, path_shots, window_size=5):
        self.path_results = path_results
        self.path_shots = path_shots
        self.window_size = window_size
        self.df = None


    def _calc_rolling_form(self, df, window=5):
        
        df_home = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'Home_xG', 'Away_xG']].copy()
        df_home.columns = ['Date', 'Team', 'Goals_S', 'Goals_C', 'xG_S', 'xG_C']

        df_away = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'Away_xG', 'Home_xG']].copy()
        df_away.columns = ['Date', 'Team', 'Goals_S', 'Goals_C', 'xG_S', 'xG_C']

        df_combined = pd.concat([df_home, df_away]).sort_values(by='Date').reset_index(drop=True)

        rolling_metrics = ['Goals_S', 'Goals_C', 'xG_S', 'xG_C']

        rolling_mean = df_combined.groupby('Team')[rolling_metrics].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

        rolling_mean.columns = [f'Forma_{col}_{window}' for col in rolling_metrics]
        df_combined = df_combined.join(rolling_mean)
        
        form_data = df_combined.drop_duplicates(subset=['Date', 'Team']).copy()
        
        return form_data, [f'Forma_{col}_{window}' for col in rolling_metrics]

    def prepare_data(self):
        # here we load data from df_epl and df_epl_shots using self.path_results and self.path_shots
        df_result = pd.read_csv(self.path_results)
        df_shots = pd.read_csv(self.path_shots)

        # get rid of odds 
        df_result= df_result.loc[:,"Date":"AR"]

        # dates in different formats in results/shots datasets
        df_result["Date"] = pd.to_datetime(df_result["Date"], dayfirst=True).dt.date
        df_shots["date"] = pd.to_datetime(df_shots["date"]).dt.date

        # obtaining xG for Home/Away team for each match 
        df_xG_grouped = df_shots.groupby(['match_id', 'h_a'])['xG'].sum().reset_index()
        df_xG_grouped.head()

        df_xG_pivot = df_xG_grouped.pivot(index="match_id", columns="h_a", values="xG").reset_index()
        df_xG_pivot.rename(columns={"h": "Home_xG", "a": "Away_xG"}, inplace=True)

        match_info = df_shots[['match_id', 'date', 'h_team', 'a_team']].drop_duplicates(subset=['match_id'])
        match_info.rename(columns={'date': 'Date', 'h_team': 'HomeTeam', 'a_team': 'AwayTeam'}, inplace=True)

        # merging dataframes to get combined df showing matches with actual scores and xG for teach team
        df_merged = pd.merge(
            df_result,
            match_info[["Date", "HomeTeam", "AwayTeam", "match_id"]],
            on=["Date", "HomeTeam", "AwayTeam"],
            how="left"
        )

        df_final = pd.merge(
            df_merged,
            df_xG_pivot[["match_id", "Home_xG", "Away_xG"]],
            on="match_id",
            how="left"
        )

        # usuwam NaN, brakuje ich w przebiegu meczu
        df_final['Date'] = pd.to_datetime(df_final['Date'])
        df_final = df_final.sort_values(by='Date').reset_index(drop=True)

        df_final.dropna(subset=['Home_xG', 'Away_xG'], inplace=True)
        df_final.reset_index(drop=True, inplace=True)

        form_data, form_cols = self._calc_rolling_form(df_final,window=self.window_size)

        df_final = pd.merge(
            df_final,
            form_data[['Date', 'Team'] + form_cols].rename(columns={'Team': 'HomeTeam'}),
            on=['Date', 'HomeTeam'],
            how='left',
            suffixes=('_M', '_H') 
        )

        df_final = pd.merge(
            df_final,
            form_data[['Date', 'Team'] + form_cols].rename(columns={'Team': 'AwayTeam'}),
            on=['Date', 'AwayTeam'],
            how='left',
            suffixes=('_H', '_A') 
        )

        # creating a new features, "AS --> Attack strength, the more the better"
        #                         "DS --> Defence strength, the less the better"
        League_Avg_Home_xG = df_final['Home_xG'].mean()
        League_Avg_Away_xG = df_final['Away_xG'].mean()
        
        # Store league averages as class attributes for prediction method
        self.league_avg_h_xg = League_Avg_Home_xG
        self.league_avg_a_xg = League_Avg_Away_xG

        df_final['Home_AS_xG'] = df_final[f'Forma_xG_S_{self.window_size}_H'] / League_Avg_Home_xG
        df_final['Away_AS_xG'] = df_final[f'Forma_xG_S_{self.window_size}_A'] / League_Avg_Away_xG
        df_final['Home_DS_xG'] = df_final[f'Forma_xG_C_{self.window_size}_H'] / League_Avg_Away_xG
        df_final['Away_DS_xG'] = df_final[f'Forma_xG_C_{self.window_size}_A'] / League_Avg_Home_xG



        self.df = df_final

    def get_prediction(self, home_team, away_team):
        """
        Predicts Expected Goals (xG) for both teams in a match.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            
        Returns:
            dict: Contains predicted xG for both teams and favorite indicator
                  Format: {
                      'home_team': str,
                      'away_team': str,
                      'home_xg': float,
                      'away_xg': float,
                      'favorite': str
                  }
        
        Raises:
            ValueError: If teams are not found or have insufficient data
        """
        try:
            # Check if data has been prepared
            if self.df is None:
                raise ValueError("Data not prepared. Please run prepare_data() first.")
            
            # Get the most recent data for home team
            home_data = self.df[self.df['HomeTeam'] == home_team]
            if home_data.empty:
                # Check if team exists as away team
                home_data = self.df[self.df['AwayTeam'] == home_team]
                if home_data.empty:
                    raise ValueError(f"Team '{home_team}' not found in the dataset.")
            
            # Get the most recent data for away team
            away_data = self.df[self.df['AwayTeam'] == away_team]
            if away_data.empty:
                # Check if team exists as home team
                away_data = self.df[self.df['HomeTeam'] == away_team]
                if away_data.empty:
                    raise ValueError(f"Team '{away_team}' not found in the dataset.")
            
            # Get the most recent row for each team (sorted by Date)
            home_latest = home_data.sort_values('Date').iloc[-1]
            away_latest = away_data.sort_values('Date').iloc[-1]
            
            # Extract Attack Strength (AS) and Defense Strength (DS)
            # Check which position the team played in their last match
            if home_latest['HomeTeam'] == home_team:
                home_as_xg = home_latest['Home_AS_xG']
                home_ds_xg = home_latest['Home_DS_xG']
            else:
                home_as_xg = home_latest['Away_AS_xG']
                home_ds_xg = home_latest['Away_DS_xG']
            
            if away_latest['AwayTeam'] == away_team:
                away_as_xg = away_latest['Away_AS_xG']
                away_ds_xg = away_latest['Away_DS_xG']
            else:
                away_as_xg = away_latest['Home_AS_xG']
                away_ds_xg = away_latest['Home_DS_xG']
            
            # Check for NaN values in key metrics
            if pd.isna(home_as_xg) or pd.isna(away_ds_xg):
                raise ValueError(f"Insufficient historical data for {home_team}.")
            if pd.isna(away_as_xg) or pd.isna(home_ds_xg):
                raise ValueError(f"Insufficient historical data for {away_team}.")
            
            # Calculate predicted xG using the formula:
            # Home xG = Home_AS_xG * Away_DS_xG * League_Avg_Home_xG
            # Away xG = Away_AS_xG * Home_DS_xG * League_Avg_Away_xG
            predicted_home_xg = home_as_xg * away_ds_xg * self.league_avg_h_xg
            predicted_away_xg = away_as_xg * home_ds_xg * self.league_avg_a_xg
            
            # Determine the favorite
            if predicted_home_xg > predicted_away_xg:
                favorite = home_team
            elif predicted_away_xg > predicted_home_xg:
                favorite = away_team
            else:
                favorite = "Draw"
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_xg': round(predicted_home_xg, 2),
                'away_xg': round(predicted_away_xg, 2),
                'favorite': favorite
            }
            
        except KeyError as e:
            raise ValueError(f"Missing required column in dataset: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    # Ten kod uruchomi się tylko gdy uruchomisz: python pipeline.py
    # Nie uruchomi się gdy zaimportujesz klasę: from pipeline import BettingEngine
    
    model = BettingEngine("datasets/premier_league_20-21.csv", "datasets/shots_premier_league_20-21.csv")
    model.prepare_data()

    print("Data has been prepared, here are 5 first rows:")
    print(model.df.head(50))