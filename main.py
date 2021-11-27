from Operations import categorical_handler
import pandas as pd

dataframe=pd.read_csv('melboun_housing_data.csv')
print(dataframe)

ch=categorical_handler(dataframe=dataframe)
updated_df=ch.remove_Categorical()
print(updated_df)
updated_df=ch.one_Hot_Encoding()
print(updated_df)
updated_df=ch.ordinal_Encoding()
print(updated_df)
