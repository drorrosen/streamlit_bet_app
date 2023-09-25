import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


class BettingBacktest:
    def __init__(self, df):
        self.df = df.dropna(subset=['Odds', 'Result'])

    def count_Loss_streaks(self):
        loss_streaks = {}
        streak_count = 0
        for result in self.df['Result']:
            if result == 'Lost':
                streak_count += 1
            else:
                if streak_count >0:
                    loss_streaks[streak_count] = loss_streaks.get(streak_count,0) + 1
                streak_count = 0
        return loss_streaks

    def backtest_sequence_xyz(self, sequence, stake):
        total_profit = 0
        seq_idx = 0

        # Add new columns for Sequence, Stakes, and PL
        self.df['Sequence'] = None
        self.df['Stakes'] = None
        self.df['PL'] = None

        for idx, row in self.df.iterrows():
            current_stake = stake * sequence[seq_idx]
            self.df.at[idx, 'Sequence'] = sequence[seq_idx]
            self.df.at[idx, 'Stakes'] = current_stake
            if row['Result'] == 'Won':
                profit = current_stake * (row['Odds'] - 1)
                self.df.at[idx, 'PL'] = profit
                seq_idx = 0
            else:
                profit = -current_stake
                self.df.at[idx, 'PL'] = profit
                seq_idx = min(len(sequence) -1, seq_idx+1)

            total_profit += profit

        return total_profit

    def bootstrap_data(self, num_samples=100):
        bootstrapped_samples = []
        for _ in range(num_samples):
            sample_df = self.df.sample(n=len(self.df), replace=True).reset_index(drop=True)
            bootstrapped_samples.append(sample_df)
        return bootstrapped_samples

    @staticmethod
    def calculate_risk_metrics(profits, confidence_level=0.95):
        sorted_profits = sorted(profits)
        var_index = int((1 - confidence_level) * len(sorted_profits))
        var_value = sorted_profits[var_index]
        cvar_value = np.mean(sorted_profits[:var_index+1])
        return var_value, cvar_value


def plot_loss_streaks(loss_streaks):
    df_loss_streaks = pd.DataFrame(list(loss_streaks.items()), columns=['Streak Length', 'Frequency'])
    fig = px.bar(df_loss_streaks, x='Streak Length', y='Frequency')
    fig.update_layout(
        title_text='Lossing streaks',
        xaxis_title='Streak Length',
        yaxis_title='Frequency',
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14))
    )

    # Show plot using Streamlit
    st.plotly_chart(fig)

def plot_simulation_profits(profits):
    fig = px.histogram(profits)
    fig.update_layout(
        title_text='Distribution of Bootstrapped Simulation Profits',
        xaxis_title='Total Profit',
        yaxis_title='Frequency',
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14))
    )

    # Show plot using Streamlit
    st.plotly_chart(fig)
