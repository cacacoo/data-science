import matplotlib.pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1074.2, 2823.1, 5922.1, 10023.8, 14231.8]

plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

plt.title("Nominal GDP")

# plt.ylable("Billions of $")
plt.show()
