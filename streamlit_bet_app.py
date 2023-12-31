import streamlit as st
import pandas as pd
from lab_simulation import *
import io
import xlsxwriter
import random
import warnings
warnings.filterwarnings('ignore')



# Set page configuration to wide layout
st.set_page_config(layout="wide")


def Intro():
    st.title('Welcome the OptimizeMyBet Wep App')

    st.write('This is a web app to optimize your bets')
    st.write('The first page is the backtesting app')
    st.write('The second page is the Lay betting app')

def page_1():
    st.title('OptimizeMyBet')

    #file uploader
    uploaded_file = st.sidebar.file_uploader('Upload a file:', type=['csv', 'xlsx'])
    if uploaded_file:
        if uploaded_file.type == 'text/csv':
            import pandas as pd

            df = pd.read_csv(uploaded_file)
            if st.checkbox('Show original data:'):
                st.dataframe(df)


        else:
            import pandas as pd
            df = pd.read_excel(uploaded_file)
            if st.checkbox('Show original data:'):
                st.dataframe(df)

        bt = BettingBacktest(df)

        columns_df = bt.df.columns
        columns_df_lower = [col.lower().strip() for col in columns_df]
        print(columns_df_lower)
        if 'sequence' in columns_df_lower:
            seq_idx = columns_df_lower.index('sequence')
            sequence_input_data = bt.df.loc[bt.df[columns_df[seq_idx]].notnull(), columns_df[seq_idx]].iloc[0]
            st.write(f"Sequence input from the file:", sequence_input_data)
        else:
            sequence_input_data = None

        counts_losses = bt.count_Loss_streaks()
        plot_loss_streaks(counts_losses)


    st.sidebar.header('User Input Parameters')


    st.divider()


    if uploaded_file:


        st.subheader('User input parameters')

        odds_options = ['User input parameter', 'Data input']
        odds_bar = st.sidebar.radio('Choose Odds', odds_options)
        if odds_bar == 'User input parameter':
            odds = st.sidebar.number_input('Odds')
            st.write(f"Odds input:", odds)
        else:
            odds = 'odds_from_file'
            st.write("Odds inputs from the file")

        stake_input = st.sidebar.number_input('Stake')
        st.write(f"Stake input:", stake_input)


        sequence_options = ['User input parameter', 'Data input']
        sequence_bar = st.sidebar.radio('Choose Sequence', sequence_options)
        if (sequence_bar == 'User input parameter'):
            sequence_input = st.sidebar.text_input('Enter Sequence here:', value="1,2,3,4,5,6")
            st.write(f"Sequence input:", sequence_input)
        elif sequence_input_data is not None:
                sequence_input = sequence_input_data
                st.write(f"Sequence input:", sequence_input)


        else:
                sequence_input = st.sidebar.text_input('No Sequence column was found in the file. Enter Sequence here:', value="1,2,3,4,5,6")
                st.write(f"Sequence input:", sequence_input)



        results = {'sequence':[float(seq) for seq in sequence_input.split(',')],
                       'stake':stake_input,
                       'odds':odds}

    st.divider()
    st.subheader('Run Simulations and backtesting to provide with more statistics')
    clicked = st.button('Begin Simulation and Backtesting:')
    if clicked:
        if uploaded_file:
            #Bootstrapping and backtesting

            bt = BettingBacktest(df)

            if results['odds'] != 'odds_from_file':
                bt.df['Odds'] = results['odds']
            if results['stake'] != 'stakes_from_file':
                bt.df['Stake'] = results['stake']
            bt.df['Stake'] = bt.df['Stake'].astype(float)


            bootstrapped_dfs = bt.bootstrap_data(num_samples=1000)
            profits = []
            for sample_df in bootstrapped_dfs:
                sample_bt = BettingBacktest(sample_df)
                profit = sample_bt.backtest_sequence_xyz(results['sequence'])
                profits.append(profit)
            plot_simulation_profits(profits)
            st.write(f"The profit average over 1000 simulations is: {np.round(np.mean(profits),2)}")
            st.write(f"The profit median over 1000 simulations is: {np.round(np.median(profits),2)}")
            st.write(f"The profit standard deviation over 1000 simulations is: {np.round(np.std(profits),2)}")

            var, cvar = BettingBacktest.calculate_risk_metrics(profits)

            st.write(f"The Value-at-Risk (VaR) on the simulated profits is: {np.round(var,2)}")
            st.write(f"The Conditional Value-at-Risk (VaR) is: {np.round(cvar,2)}")


    st.divider()

    st.subheader('Running Backtesting and creating a file to download')

    if uploaded_file:
        clicked = st.button('Begin Backtesting and creating downloaded file')
        if clicked:
            bt = BettingBacktest(df)

            if results['odds'] != 'odds_from_file':
                bt.df['Odds'] = results['odds']
            if results['stake'] != 'stakes_from_file':
                bt.df['Stake'] = results['stake']
            bt.df['Stake'] = bt.df['Stake'].astype(float)



            result = bt.backtest_sequence_xyz(results['sequence'])
            bt.df['Stakes'] = results['stake']
            bt.df = bt.df[['Time', 'Race', 'Selection', 'BetType', 'Odds', 'Sequence', 'Stake','Result', 'PL']]
            st.write(bt.df)
            st.write('Total PL: ', np.round(bt.df['PL'].sum(),2))

            # Create a download button

            # Write the DataFrame to a BytesIO object
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                bt.df.to_excel(writer, sheet_name='Sheet1', index=False)

            st.download_button(
                label="Download Excel File",
                data=output,
                file_name="my_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )

    st.divider()

    st.subheader('Backtesting Sequence Finder')


    if uploaded_file:
        clicked = st.button('OptimizeMyBet Sequence Finder')
        if clicked:
            bt = BettingBacktest(df)
            if results['odds'] != 'odds_from_file':
             bt.df['Odds'] = results['odds']
            if results['stake'] != 'stakes_from_file':
                bt.df['Stake'] = results['stake']
                bt.df['Stake'] = bt.df['Stake'].astype(float)

            # Initialize the progress bar
            progress_bar = st.progress(0)
            # Initialize a random sequence
            initial_sequence = generate_limited_random_sequence()

            # Perform hill climbing to find the best sequence using backtesting
            best_sequence_backtest, best_profit_backtest = hill_climb_with_backtest(initial_sequence, backtest_instance=bt)
            st.write(f"Best sequence:, :blue[{','.join([str(seq) for seq in best_sequence_backtest])}]")
            st.write(f"Best PL: :blue[{np.round(best_profit_backtest,2)}]")



def page_2():
    st.title('OptimizeMyBet')

    #file uploader
    uploaded_file = st.sidebar.file_uploader('Upload a file:', type=['csv', 'xlsx'])
    if uploaded_file:
        if uploaded_file.type == 'text/csv':
            import pandas as pd

            df = pd.read_csv(uploaded_file)
            if st.checkbox('Show original data:'):
                st.dataframe(df)


        else:
            import pandas as pd
            df = pd.read_excel(uploaded_file)
            if st.checkbox('Show original data:'):
                st.dataframe(df)

        bt = LayBettingBacktest(df)

        columns_df = bt.df.columns
        columns_df_lower = [col.lower().strip() for col in columns_df]
        if 'sequence' in columns_df_lower:
            seq_idx = columns_df_lower.index('sequence')
            sequence_input_data = bt.df.loc[bt.df[columns_df[seq_idx]].notnull(), columns_df[seq_idx]].iloc[0]
            st.write(f"Sequence input from the file:", sequence_input_data)
        else:
            sequence_input_data = None

        counts_losses = bt.count_Loss_streaks()
        plot_loss_streaks(counts_losses)


    st.sidebar.header('User Input Parameters')


    st.divider()


    if uploaded_file:


        st.subheader('User input parameters')

        odds_options = ['User input parameter', 'Data input']
        odds_bar = st.sidebar.radio('Choose Odds', odds_options)
        if odds_bar == 'User input parameter':
            odds = st.sidebar.number_input('Odds')
            st.write(f"Odds input:", odds)
        else:
            odds = 'odds_from_file'
            st.write("Odds inputs from the file")

        stake_input = st.sidebar.number_input('Stake')
        st.write(f"Stake input:", stake_input)


        sequence_options = ['User input parameter', 'Data input']
        sequence_bar = st.sidebar.radio('Choose Sequence', sequence_options)
        if (sequence_bar == 'User input parameter'):
            sequence_input = st.sidebar.text_input('Enter Sequence here:', value="1,2,3,4,5,6")
            st.write(f"Sequence input:", sequence_input)
        elif sequence_input_data is not None:
            sequence_input = sequence_input_data
            st.write(f"Sequence input:", sequence_input)


        else:
            sequence_input = st.sidebar.text_input('No Sequence column was found in the file. Enter Sequence here:', value="1,2,3,4,5,6")
            st.write(f"Sequence input:", sequence_input)



        results = {'sequence':[float(seq) for seq in sequence_input.split(',')],
                   'stake':stake_input,
                   'odds':odds}

    st.divider()
    st.subheader('Run Simulations and lay betting to provide with more statistics')
    clicked = st.button('Begin Simulation and Backtesting:')
    if clicked:
        if uploaded_file:
            #Bootstrapping and backtesting

            bt = LayBettingBacktest(df)

            if results['odds'] != 'odds_from_file':
                bt.df['Odds'] = results['odds']
            if results['stake'] != 'stakes_from_file':
                bt.df['Stake'] = results['stake']
            bt.df['Stake'] = bt.df['Stake'].astype(float)


            bootstrapped_dfs = bt.bootstrap_data(num_samples=1000)
            profits = []
            for sample_df in bootstrapped_dfs:
                sample_bt = LayBettingBacktest(sample_df)
                profit = sample_bt.backtest_lay_sequence(results['sequence'])
                profits.append(profit)
            plot_simulation_profits(profits)
            st.write(f"The profit average over 1000 simulations is: {np.round(np.mean(profits),2)}")
            st.write(f"The profit median over 1000 simulations is: {np.round(np.median(profits),2)}")
            st.write(f"The profit standard deviation over 1000 simulations is: {np.round(np.std(profits),2)}")

            var, cvar = LayBettingBacktest.calculate_risk_metrics(profits)

            st.write(f"The Value-at-Risk (VaR) on the simulated profits is: {np.round(var,2)}")
            st.write(f"The Conditional Value-at-Risk (VaR) is: {np.round(cvar,2)}")


    st.divider()

    st.subheader('Running lay betting and creating a file to download')

    if uploaded_file:
        clicked = st.button('Begin Backtesting and creating downloaded file')
        if clicked:
            bt = LayBettingBacktest(df)

            if results['odds'] != 'odds_from_file':
                bt.df['Odds'] = results['odds']
            if results['stake'] != 'stakes_from_file':
                bt.df['Stake'] = results['stake']
            bt.df['Stake'] = bt.df['Stake'].astype(float)



            result = bt.backtest_lay_sequence(results['sequence'])
            bt.df['Stakes'] = results['stake']
            bt.df = bt.df[['Time', 'Race', 'Selection', 'BetType', 'Odds', 'Sequence', 'Stake','Result', 'PL']]
            st.write(bt.df)
            st.write('Total PL: ', np.round(bt.df['PL'].sum(),2))

            # Create a download button

            # Write the DataFrame to a BytesIO object
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                bt.df.to_excel(writer, sheet_name='Sheet1', index=False)

            st.download_button(
                label="Download Excel File",
                data=output,
                file_name="my_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )

    st.divider()

    st.subheader('Lay Betting Sequence Finder')


    if uploaded_file:
        clicked = st.button('OptimizeMyBet Sequence Finder')
        if clicked:
            bt = LayBettingBacktest(df)
            if results['odds'] != 'odds_from_file':
                bt.df['Odds'] = results['odds']
            if results['stake'] != 'stakes_from_file':
                bt.df['Stake'] = results['stake']
                bt.df['Stake'] = bt.df['Stake'].astype(float)

            # Initialize the progress bar
            progress_bar = st.progress(0)
            # Initialize a random sequence
            initial_sequence = generate_limited_random_sequence()

            # Perform hill climbing to find the best sequence using backtesting
            best_sequence_backtest, best_profit_backtest = hill_climb_lay_betting(initial_sequence, backtest_instance=bt)
            st.write(f"Best sequence:, :blue[{','.join([str(seq) for seq in best_sequence_backtest])}]")
            st.write(f"Best PL: :blue[{np.round(best_profit_backtest,2)}]")




page_names_to_funcs = {
    "Introduction": Intro,
    "OptimizeMyBet": page_1,
    "OptimizeMyLayBet": page_2,

}

dashboard_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[dashboard_name]()
#%%
