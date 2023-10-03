import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import random
from tqdm import tqdm


class BettingBacktest:
    def __init__(self, df):
        self.df = df.dropna(subset=['Odds', 'Result', 'Stake'])

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

    def backtest_sequence_xyz(self, sequence):
        total_profit = 0
        seq_idx = 0

        # Add new columns for Sequence, Stakes, and PL
        self.df['Sequence'] = None
        #self.df['Stakes'] = None
        self.df['PL'] = None

        for idx, row in self.df.iterrows():
            current_stake = float(row['Stake']) * sequence[seq_idx]
            self.df.at[idx, 'Sequence'] = sequence[seq_idx]
            #self.df.at[idx, 'Stakes'] = current_stake
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


def generate_limited_random_sequence(min_length=3, max_length=15, value_ranges=[(1, 20), (25, 100, 5)]):
    """
    Generate a random sequence of integers within specified value ranges and varying length.
    The first number in the sequence is always 1.

    Parameters:
    - min_length: Minimum length of the sequence
    - max_length: Maximum length of the sequence
    - value_ranges: List of tuples specifying the ranges for random values.
                    Each tuple can be (min_value, max_value) or (min_value, max_value, step)

    Returns:
    A list of integers representing the sequence
    """
    # The first number in the sequence must be 1
    sequence = [1]

    # Determine the length of this sequence
    sequence_length = random.randint(min_length, max_length)

    # Generate the remaining numbers in the sequence
    for _ in range(sequence_length - 1):
        # Randomly pick a value range
        value_range = random.choice(value_ranges)

        if len(value_range) == 2:
            # Generate a random value within this range
            sequence.append(random.randint(value_range[0], value_range[1]))
        else:
            # Generate a random value within this range with the specified step
            sequence.append(random.randrange(value_range[0], value_range[1] + 1, value_range[2]))

    return sequence



# Modify the hill climbing function to use the backtest_sequence_xyz method for evaluation
def hill_climb_with_backtest(initial_sequence, iterations=10000, backtest_instance=None):
    """
    Perform hill climbing to find the sequence that maximizes profit using backtesting.

    Parameters:
    - initial_sequence: The initial betting sequence
    - iterations: The number of iterations to perform
    - backtest_instance: An instance of the BettingBacktest class for backtesting sequences

    Returns:
    The sequence that results in the highest profit according to backtesting
    """
    # Initialize variables
    current_sequence = initial_sequence
    best_profit = backtest_instance.backtest_sequence_xyz(current_sequence) if backtest_instance else 0

    progress_text = st.empty()
    # Perform hill climbing
    for i in tqdm(range(iterations), desc='Processing'):
        # Generate a neighbor by perturbing the current sequence
        neighbor_sequence = current_sequence[:]
        action = random.choice(["add", "remove", "change"])

        if action == "add":
            neighbor_sequence.append(random.randint(1, 100))
        elif action == "remove" and len(neighbor_sequence) > 1:
            del neighbor_sequence[random.randint(1, len(neighbor_sequence) - 1)]
        elif action == "change" and len(neighbor_sequence) > 1:
            neighbor_sequence[random.randint(1, len(neighbor_sequence) - 1)] = random.randint(1, 100)

        # Evaluate the neighbor using backtest_sequence_xyz
        neighbor_profit = backtest_instance.backtest_sequence_xyz(neighbor_sequence) if backtest_instance else 0

        # If the neighbor is better, move to the neighbor state
        if neighbor_profit > best_profit:
            current_sequence = neighbor_sequence
            best_profit = neighbor_profit

        progress_text.text(f"Progress: {np.round((i+1)/iterations*100,2)}%")
    return current_sequence, best_profit
