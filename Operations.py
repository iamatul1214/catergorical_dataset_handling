import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, max_error

class categorical_handler:
    def __init__(self,dataframe):
        self.dataframe=dataframe

    def return_categorical_features(self):
        """
        This method will return the categorical features in the passed dataset.
        :return:
        """
        try:
            df=self.dataframe
            cat=(df.dtypes=='object')
            categorical_features=list(cat[cat].index)
            print("categorical features={}".format(categorical_features))
            return categorical_features
        except Exception as e:
            raise Exception("Error occured while returning the categorical features from the dataset{1}".format(str(e)))

    def remove_Categorical(self):
        """
        This method will remove the categorical features from the dataset passed in argument
        """
        try:
            cat_features = self.return_categorical_features()
            df=self.dataframe
            updated_df=df.drop(columns=cat_features)
            return updated_df
        except Exception as e:
            remove_categorical_exception="failed to drop the categorical features{0} from the dataset".format(cat_features)
            raise Exception(remove_categorical_exception+"More details={0}".format(str(e)))

    def ordinal_Encoding(self):
        """
        This method will encode the categorical values with their frequency of occurances in the column.
        """

        try:
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
        except Exception as e:
            raise Exception("Error occured while performing ordinal encoding for the categorical dataset{0}".format(str(e)))

    def one_Hot_Encoding(self):
        """
        This method will return the one hot encoded features with n-1 features. Where n is total number of features in the passed dataset
        """
        try:
            cat_features=self.return_categorical_features()
            df=self.dataframe
            one_hot_encode=pd.get_dummies(df[cat_features], drop_first=True)
            df=df.drop(columns=cat_features)
            frames=[df,one_hot_encode]
            updated_df=pd.concat(frames,axis=1)
            return updated_df
        except Exception as e:
            raise Exception("Error occured while  performing one hot encoding{0}".format(str(e)))


    def mean_Encoding_with_Target_Column(self,categorical_features,label):
        try:
            """
                This method will encode the categorical feature into numerical by considering it's mean with respect to label
                column entered.
                ********************************************************************
                categorical_features= List of categorical features in your dataset
                label=The label column or the target column of your dataset
                ********************************************************************
                example-
                categorical_features=['column_name1','column_name2']
                label=result
            """
            categorical_features=categorical_features
            target_column=label
            df=self.dataframe
            for i in range(len(categorical_features)):
                encoded_mean_dict=df.groupby([categorical_features[i]]).mean()[target_column].to_dict()
                df[categorical_features[i]+"_mean_encoded"]=df[categorical_features[i]].map(encoded_mean_dict)
                df=df.drop(columns=categorical_features[i])
            return df
        except Exception as e:
            raise Exception("Error occured while performing mean encoding with respect to target column{0}".format(str(e)))






