import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, max_error

class categorical_handler:
    def __init__(self,dataframe):
        self.dataframe=dataframe

    def return_categorical_features(self):
        df=self.dataframe
        cat=(df.dtypes=='object')
        categorical_features=list(cat[cat].index)
        print("categorical features={}".format(categorical_features))
        return categorical_features

    def remove_Categorical(self):
        cat_features = self.return_categorical_features()
        df=self.dataframe
        updated_df=df.drop(columns=cat_features)
        return updated_df

    def ordinal_Encoding(self):
        cat_features = self.return_categorical_features()
        cat_features_ordinal=[]
        df=self.dataframe

        for i in range(len(cat_features)):
            uniques=df[cat_features[i]].unique()
            values=range(len(uniques))
            keys=uniques
            ordinal_dict=zip(keys,values)
            cat_features_ordinal.append(ordinal_dict)

        for i in range(len(cat_features_ordinal)):
            df[cat_features[i]]=df[cat_features[i]].replace(cat_features_ordinal[i])
        updated_df=pd.DataFrame(df)
        return updated_df

    def one_Hot_Encoding(self,):
        cat_features=self.return_categorical_features()
        df=self.dataframe
        one_hot_encode=pd.get_dummies(df[cat_features], drop_first=True)
        df=df.drop(columns=cat_features)
        frames=[df,one_hot_encode]
        updated_df=pd.concat(frames,axis=1)
        return updated_df





