import pandas as pd

data = pd.read_csv('data.csv')

count = 0
for i in data['Taken/Not-Taken']:
    if(i== count):
        count+=1
print(count)
print("Taken Test")





