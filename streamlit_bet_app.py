import streamlit as st
import pandas as pd
from lab_simulation import *


# Set page configuration to wide layout
st.set_page_config(layout="wide")


st.title('Sequence Testing App')

#file uploader
uploaded_file = st.sidebar.file_uploader('Upload a file:', type=['csv', 'xlsx'])
if uploaded_file:
    if uploaded_file.type == 'text/csv':
        import pandas as pd

        df = pd.read_csv(uploaded_file)
        if st.checkbox('Show data:'):
            st.write(df)

    else:
        import pandas as pd
        df = pd.read_excel(uploaded_file)
        if st.checkbox('Show data:'):
            st.write(df)

st.sidebar.header('User Input Parameters')


st.divider()


if uploaded_file:
    bt = BettingBacktest(df)
    loss_streaks = bt.count_Loss_streaks()
    plot_loss_streaks(loss_streaks)


    def user_input_features():
        sequence = st.sidebar.text_input('Enter Sequence here:', value="1,2,3,4,5,6")

        return sequence

    sequence_input = user_input_features()

    st.subheader('User input parameters')

    odds_options = ['User input parameter', 'Data input']
    odds_bar = st.sidebar.radio('Choose Odds', odds_options)
    if odds_bar == 'User input parameter':
        odds = st.sidebar.slider('Odds', 0.0,10.0,1.5,0.1)
        st.write(f"Odds input:", odds)
    else:
        odds = 'odds_from_file'
        st.write("Odds inputs from the file")

    st.write(f'Chosen Sequence: {sequence_input}')
    results = {'sequence':[float(seq) for seq in sequence_input.split(',')],
               'odds':odds}

st.divider()

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
            profit = sample_bt.backtest_sequence_xyz(results['sequence'])
            profits.append(profit)
        plot_simulation_profits(profits)
        st.write(f"The profit average over 1000 simulations is: {np.round(np.mean(profits),2)}")
        st.write(f"The profit median over 1000 simulations is: {np.round(np.std(profits),2)}")
        st.write(f"The profit standard deviation over 1000 simulations is: {np.round(np.median(profits),2)}")

        var, cvar = BettingBacktest.calculate_risk_metrics(profits)

        st.write(f"The Value-at-Risk (VaR) on the simulated profits is: {np.round(var,2)}")
        st.write(f"The Conditional Value-at-Risk (VaR) is: {np.round(cvar,2)}")


else:
        st.write("You didn't upload a dataset")

st.divider()