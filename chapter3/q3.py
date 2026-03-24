from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def fetch_titanic_data():
    tarball_path = Path('datasets/titanic.tgz')
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://homl.info/titanic.tgz'
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path='datasets', filter='data')
    return (pd.read_csv(Path('datasets/titanic/test.csv')), pd.read_csv(Path('datasets/titanic/train.csv')))

test_data, train_data = fetch_titanic_data()

# print(train_data.info())

# <class 'pandas.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#      Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   PassengerId  891 non-null    int64  
#  1   Survived     891 non-null    int64   -> binary variable expressed in a 0 and 1
#  2   Pclass       891 non-null    int64   -> 1, 2 or 3
#  3   Name         891 non-null    str     -> a string of name, not sure how that would be helpful for the algorithm 
#  4   Sex          891 non-null    str     -> this is a binary value, need to transform 
#  5   Age          714 non-null    float64 -> has missing data, maybe worth extrapolating it somehow
#  6   SibSp        891 non-null    int64   -> not sure what this is, tbh
#  7   Parch        891 non-null    int64   -> not sure what this is either
#  8   Ticket       891 non-null    str     -> it may be an important parameter, but have no idea how to transform a string into a number
#  9   Fare         891 non-null    float64 -> may be useful
#  10  Cabin        204 non-null    str     -> has a lot of missing data (77% of data is missing, maybe delete it all together), but cabin number may be valuable
#  11  Embarked     889 non-null    str     -> has some missing data, looks like it is a categorial value (-> replace with a head map)
# dtypes: float64(2), int64(5), str(5)
# memory usage: 83.7 KB
# None

# print(train_data['Sex'].value_counts()) -> transform to binary

# Sex
# male      577
# female    314
# Name: count, dtype: int64

# print(train_data['Embarked'].value_counts()) -> transform to 1, 2 or 3

# Embarked
# S    644
# C    168
# Q     77
# Name: count, dtype: int64

# print(train_data.describe())

#        PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699113    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071   14.526507    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000    0.416700    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200