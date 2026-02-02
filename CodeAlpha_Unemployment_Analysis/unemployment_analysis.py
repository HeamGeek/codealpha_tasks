# --------------------------------
# Unemployment Analysis with Python
# CodeAlpha Internship – Task 2
# --------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load dataset ----
df = pd.read_csv("data/Unemployment_Rate_upto_11_2020.csv")

print("Preview of data:")
print(df.head())

# remove BOMs/unwanted chars and strip surrounding whitespace while handles leading/trailing spaces or different casing
df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

date_col = None
for col in df.columns:
	if col.strip().lower().startswith("date"):
		date_col = col
		break

if date_col is None:
	raise KeyError("No Date-like column found in CSV. Columns: " + ", ".join(df.columns))

# strip Date values to handle leading spaces in the CSV
df[date_col] = df[date_col].astype(str).str.strip()
df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors="coerce")
if date_col != "Date":
	df.rename(columns={date_col: "Date"}, inplace=True)

# ---- Data overview ----
print("\nData Info:")
print(df.info())

print("\nMissing values by column:")
print(df.isnull().sum())

# ---- NATIONAL TREND: Unemployment Rate over time ----
plt.figure(figsize=(12, 5))
national_trend = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()

sns.lineplot(
    x="Date",
    y="Estimated Unemployment Rate (%)",
    data=national_trend
)

plt.title("Unemployment Rate in India Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/unemployment_trend.png", dpi=150)
plt.show()

# ---- COVID-19 IMPACT ----
covid_start = "2020-01-01"
covid_data = df[df["Date"] >= covid_start]

plt.figure(figsize=(12, 5))
sns.lineplot(x="Date", y="Estimated Unemployment Rate (%)", data=covid_data)
plt.title("Unemployment Rate During and After COVID-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/unemployment_covid.png", dpi=150)
plt.show()

# ---- REGION-WISE ANALYSIS ----
plt.figure(figsize=(14, 8))
sns.boxplot(x="Region", y="Estimated Unemployment Rate (%)", data=df)
plt.title("Unemployment Rate by Region (State)")
plt.xlabel("State")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("results/unemployment_by_region.png", dpi=150)
plt.show()
