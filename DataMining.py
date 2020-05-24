import csv
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DataPoint:
    def __init__(self, name, industry, inception, employees, state, city, revenue, expenses, profit, growth):
        self.name = name
        self.industry = industry
        self.inception = inception
        self.employees = employees
        self.state = state 
        self.city = city
        self.revenue = revenue
        self.expenses = expenses.replace(',', '').replace(' Dollars', '')
        self.profit = profit
        self.growth = growth

futureList = []

with open('Future_CSV.csv', newline='') as csvfile:
    csvReader = csv.DictReader(csvfile)
    for row in csvReader:
        futureList.append(DataPoint(row['Name'], row['Industry'], row['Inception'], row['Employees'], row['State'], row['City'], row['Revenue'], row['Expenses'], row['Profit'], row['Growth']))

dataFrame = pd.DataFrame([t.__dict__ for t in futureList])

dataFrame['employees'] = pd.to_numeric(dataFrame['employees'], errors='coerce')
dataFrame['revenue'] = pd.to_numeric(dataFrame['revenue'], errors='coerce')
dataFrame['expenses'] = pd.to_numeric(dataFrame['expenses'], errors='coerce')
dataFrame['profit'] = pd.to_numeric(dataFrame['profit'], errors='coerce')
dataFrame['growth'] = pd.to_numeric(dataFrame['growth'], errors='coerce')

print(dataFrame.describe(include='all'))

quantilesAll = dataFrame.quantile([.25, .75], axis=0)

lowQ = dataFrame['employees'].quantile(0.25)
highQ = dataFrame['employees'].quantile(0.75)
funcH = (highQ - lowQ) * 9
filteredData = dataFrame[(dataFrame['employees'] > lowQ-funcH) & (dataFrame['employees'] < highQ+funcH)]

outliersRem = pd.merge(dataFrame, filteredData, how='inner')
# print("---employees-----------------------------------------------------------")
# print(filteredData.describe())

lowQ = dataFrame['revenue'].quantile(0.25)
highQ = dataFrame['revenue'].quantile(0.75)
funcH = (highQ - lowQ) * 3
filteredData = dataFrame[(dataFrame['revenue'] > lowQ-funcH) & (dataFrame['revenue'] < highQ+funcH)]

outliersRem = pd.merge(outliersRem, filteredData, how='inner')
# print("---revenue-----------------------------------------------------------")
# print(filteredData.describe())

lowQ = dataFrame['expenses'].quantile(0.25)
highQ = dataFrame['expenses'].quantile(0.75)
funcH = (highQ - lowQ) * 3
filteredData = dataFrame[(dataFrame['expenses'] > lowQ-funcH) & (dataFrame['expenses'] < highQ+funcH)]

outliersRem = pd.merge(outliersRem, filteredData, how='inner')
# print("---expenses-----------------------------------------------------------")
# print(filteredData.describe())

lowQ = dataFrame['profit'].quantile(0.25)
highQ = dataFrame['profit'].quantile(0.75)
funcH = (highQ - lowQ) * 3
filteredData = dataFrame[(dataFrame['profit'] > lowQ-funcH) & (dataFrame['profit'] < highQ+funcH)]

outliersRem = pd.merge(outliersRem, filteredData, how='inner')
# print("---profit-----------------------------------------------------------")
# print(filteredData.describe())

lowQ = dataFrame['growth'].quantile(0.25)
highQ = dataFrame['growth'].quantile(0.75)
funcH = (highQ - lowQ) * 3
filteredData = dataFrame[(dataFrame['growth'] > lowQ-funcH) & (dataFrame['growth'] < highQ+funcH)]

outliersRem = pd.merge(outliersRem, filteredData, how='inner')
# print("---growth-----------------------------------------------------------")
# print(filteredData.describe())

# print("------------------------------------------------------------------")
# print("---all-----------------------------------------------------------")
# print(outliersRem.describe())


minMaxScale = MinMaxScaler()
minMaxAfter = outliersRem.copy()
minMaxAfter[['employees', 'revenue', 'expenses', 'profit', 'growth']] = minMaxScale.fit_transform(minMaxAfter[['employees', 'revenue', 'expenses', 'profit', 'growth']])
# plt.bar(minMaxAfter['industry'], minMaxAfter['revenue'])
# plt.xlabel("Industry")
# plt.ylabel("Revenue")
# plt.show()

standardScale = StandardScaler()
standardAfter = outliersRem.copy()
standardAfter[['employees', 'revenue', 'expenses', 'profit', 'growth']] = standardScale.fit_transform(standardAfter[['employees', 'revenue', 'expenses', 'profit', 'growth']])
# plt.bar(standardAfter['industry'], standardAfter['revenue'])
# plt.xlabel("Industry")
# plt.ylabel("Revenue")
# plt.axhline(0, color='k')
# plt.show()


# plt.boxplot(dataFrame['employees'])
# plt.xlabel("Employees")
# plt.ylabel("Amount")
# plt.show()

# plt.bar(outliersRem['industry'].value_counts().index, outliersRem['industry'].value_counts())
# plt.xlabel("Industry")
# plt.ylabel("Frequency")
# plt.show()

# plt.hist2d(outliersRem['revenue'], outliersRem['profit'])
# plt.xlabel("Revenue")
# plt.ylabel("Profit")
# plt.colorbar()
# plt.show()

# plt.hist2d(outliersRem['revenue'], outliersRem['growth'])
# plt.xlabel("Revenue")
# plt.ylabel("Growth")
# plt.colorbar()
# plt.show()

# sns.pairplot(x_vars=['revenue'], y_vars=['expenses'], data=outliersRem, hue='industry', height=5)
# plt.xlabel("Revenue")
# plt.ylabel("Expenses")
# plt.show()


embedding = MDS(n_components=2)

X_transformed = embedding.fit_transform(standardAfter[['employees', 'revenue', 'expenses', 'profit', 'growth']][:487])
MDSDataFrame = pd.DataFrame(data=X_transformed, columns=['component 1', 'component 2'])
MDSDataFrame['industry'] = standardAfter['industry']
sns.pairplot(x_vars=['component 1'], y_vars=['component 2'], data=MDSDataFrame, hue='industry', height=5)
plt.show()


print(standardAfter.corr(method='pearson'))