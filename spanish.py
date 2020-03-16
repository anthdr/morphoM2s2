import pandas as pd
import numpy as np
data = pd.read_csv("origin/spanish-paradigm.csv")

data['regular'] = data['class'].apply(lambda x: '1' if 'regular' in x else '0')
data['last'] = data['stem'].str.strip().str[-1]
data['last2'] = data['stem'].str.strip().str[-2]
data['last3'] = data['stem'].str.strip().str[-3]
data['last4'] = data['stem'].str.strip().str[-4]
data['first'] = data['stem'].str.strip().str[0]
data['first2'] = data['stem'].str.strip().str[1]
data['first3'] = data['stem'].str.strip().str[2]
data['first4'] = data['stem'].str.strip().str[3]

print(data.head(5))
print(data.tail(5))

data.to_csv(r'spanish-paradigm.csv', index = False)