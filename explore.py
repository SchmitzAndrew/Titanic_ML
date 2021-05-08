import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def training_data():
    training_file_path = "data/train.csv"

    titanic_data = pd.read_csv(training_file_path)
    print(titanic_data.describe())
    print(titanic_data.describe().columns)
    print(titanic_data.head().columns)

    # Graphing Numerical and Categorical Data

    df_numerical = titanic_data[['Age', 'SibSp', 'Parch', 'Fare']]
    df_categorical = titanic_data[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

    for d in df_numerical.columns:
        plt.hist(df_numerical[d])
        plt.title(d)
        plt.show()

    for d in df_categorical.columns:
        sns.barplot(x = df_categorical[d].value_counts().index, y=df_categorical[d].value_counts()).set_title(d) #x/y must be defined
        plt.show()

def gender_data():
    gender_file_path = "data/gender_submission.csv"
    gender_data = pd.read_csv(gender_file_path)

    print(gender_data.describe().columns)
    print(gender_data.describe())

    print(gender_data.head())


#training_data()
gender_data()
