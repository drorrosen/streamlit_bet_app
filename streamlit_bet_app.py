import streamlit as st
import pandas as pd
from lab_simulation import *
import io
import xlsxwriter


# Set page configuration to wide layout
st.set_page_config(layout="wide")


st.title('Sequence Testing App')

#file uploader
uploaded_file = st.sidebar.file_uploader('Upload a file:', type=['csv', 'xlsx'])
if uploaded_file:
    if uploaded_file.type == 'text/csv':
        import pandas as pd

        df = pd.read_csv(uploaded_file)
        if st.checkbox('Show original data:'):
            st.write(df)

    else:
        import pandas as pd
        df = pd.read_excel(uploaded_file)
        if st.checkbox('Show original data:'):
            st.write(df)

st.sidebar.header('User Input Parameters')


st.divider()


if uploaded_file:



    def user_input_features():
        sequence = st.sidebar.text_input('Enter Sequence here:', value="1,2,3,4,5,6")
        stake = st.sidebar.number_input('Enter base Stake:', value=10)

        return sequence, stake

    sequence_input, stake_input = user_input_features()

    st.subheader('User input parameters')

    odds_options = ['User input parameter', 'Data input']
    odds_bar = st.sidebar.radio('Choose Odds', odds_options)
    if odds_bar == 'User input parameter':
        odds = st.sidebar.number_input('Odds')
        st.write(f"Odds input:", odds)
    else:
        odds = 'odds_from_file'
        st.write("Odds inputs from the file")

    st.write(f'Chosen Stake:', stake_input)
    st.write(f'Chosen Sequence: {sequence_input}')
    results = {'sequence':[float(seq) for seq in sequence_input.split(',')],
               'stake':stake_input,
               'odds':odds}

st.divider()
st.subheader('Run Simulations and backtesting to provide with more statistics')
clicked = st.button('Begin Simulation and Backtesting:')
if clicked:
    if uploaded_file:
        #Bootstrapping and backtesting
        if results['odds'] != 'odds_from_file':
            df['Odds'] = results['odds']

        bt = BettingBacktest(df)
        bootstrapped_dfs = bt.bootstrap_data(num_samples=1000)
        profits = []
        for sample_df in bootstrapped_dfs:
            sample_bt = BettingBacktest(sample_df)
            profit = sample_bt.backtest_sequence_xyz(results['sequence'], results['stake'])
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
        result = bt.backtest_sequence_xyz(results['sequence'], results['stake'])
        bt.df = bt.df[['Time', 'Race', 'Selection', 'BetType', 'Odds', 'Sequence', 'Stakes', 'PL']]
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
