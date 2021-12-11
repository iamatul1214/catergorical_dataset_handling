from CatDealer.CatHandler import categorical_handler
import pandas as pd

dataframe=pd.read_csv('Titanic.csv')
print(dataframe)

ch=categorical_handler(dataframe=dataframe)
updated_df=ch.remove_Categorical()
print(updated_df)
updated_df=ch.one_Hot_Encoding()
print(updated_df)
updated_df=ch.ordinal_Encoding()
print(updated_df)
updated_df=ch.mean_Encoding_with_Target_Column(categorical_features=['Cabin','Sex'],label='Survived')
print(updated_df)
