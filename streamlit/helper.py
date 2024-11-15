import pandas as pd

data = pd.read_csv('data/train_AIC.csv')
columns = data.iloc[:, :-1].columns
for column in columns:
    print(f'st.number_input(label="{column}")')
