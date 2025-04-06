import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="protean-unity-452919-h4-5ef9777c8a6e.json" # lokalizacja pobranego klucza z punktu 1.4.
client = bigquery.Client()

Entry: int = 10000
#agregacja na 0 (?)
query = (
    f'select * from bigquery-public-data.covid19_open_data.covid19_open_data limit {Entry}')
query_job = client.query(query)
query_result = query_job.result()
df = query_result.to_dataframe()

columns = ['new_confirmed_age_0', 'new_confirmed_male', 'search_trends_abdominal_obesity', 'cumulative_recovered_age_9']
column_numbers = [df.columns.get_loc(col) for col in columns]
for col in df.columns[::-1]:
    col_idx = df.columns.get_loc(col)
    if (column_numbers[0] <= col_idx and col_idx <= column_numbers[1]-1) \
    or (column_numbers[2] <= col_idx and col_idx <= column_numbers[3]):
        threshold = Entry
    else:
        threshold = -1
    if df[col].count() <= threshold:
        df = df.drop(columns=[col])
print(f"Liczba kolumn po usunięciu: {df.shape[1]}")

df = df.drop_duplicates()
df = df.dropna(thresh=len(df.columns)*0.04) #Jeżeli 4% wiersza nie jest wypełnione wyrzucić
print(f"Liczba wierszy po usunięciu: {df.shape[0]}")

# for dtype in df.dtypes.unique():
#     if dtype not in ['float64', 'Int64']:
#         columns_of_dtype = df.select_dtypes(include=[dtype])
#         for column in columns_of_dtype:
#             sample_value = columns_of_dtype[column].dropna().unique()[:4] if not \
#             columns_of_dtype[column].dropna().empty else None
#             print(f"    Próbka z kolumny '{column}' typu '{dtype}': {sample_value}")
columns = ["new_tested", "life_expectancy", "mobility_retail_and_recreation", "mobility_grocery_and_pharmacy",  \
           "mobility_parks", "mobility_transit_stations", "mobility_workplaces", "mobility_residential",        \
           "population_largest_city", "population_clustered", "human_capital_index", "area_rural_sq_km",        \
           "area_urban_sq_km", "adult_male_mortality_rate", "adult_female_mortality_rate",                      \
           "pollution_mortality_rate", "comorbidity_mortality_rate"]
for col in df.columns:
    if col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
print("Przekonwertowano stringi liczb na liczby")
# for dtype in df.dtypes.unique():
#     if dtype not in ['float64', 'Int64']:
#         columns_of_dtype = df.select_dtypes(include=[dtype])
#         for column in columns_of_dtype:
#             sample_value = columns_of_dtype[column].dropna().unique()[:4] if not \
#             columns_of_dtype[column].dropna().empty else None
#             print(f"    Próbka z kolumny '{column}' typu '{dtype}': {sample_value}")

# columns_with_negatives = []
# for dtype in df.dtypes.unique():
#     if dtype not in ['object', 'dbdate']:
#         columns_of_dtype = df.select_dtypes(include=[dtype])
#         for col in columns_of_dtype:
#             if (df[col] < 0).any():
#                 columns_with_negatives.append(col)
# print("Kolumny z wartościami ujemnymi:", columns_with_negatives)
columns = ["population_age_80_and_older"]
for col in columns:
    mask = df[col] < 0
    df[col] = df[col].where(~mask, np.nan)
    print(f"Kolumna: {col}, Liczba zmienionych wartości: {mask.sum()}")

#4 ----------------------------------------------------------------------------
#print(df.columns.to_list())
#4.1 #agregacja tylko dla 0
columns = ["date", "country_name", "aggregation_level", "population", "population_density",                         \
           "human_development_index", "gdp_per_capita_usd", "area_sq_km", "population_age_00_09",                   \
           "population_age_10_19", "population_age_20_29", "population_age_30_39", "population_age_40_49",          \
           "population_age_50_59", "population_age_60_69", "population_age_70_79", "population_age_80_and_older",   \
           "smoking_prevalence", "diabetes_prevalence", "human_capital_index", "life_expectancy"]
countries_df = df[columns]
countries_df = countries_df.sort_values(by='date', ascending=False)
countries_df = countries_df.drop_duplicates(subset='country_name', keep='first')
countries_df.to_csv('0401_countries.csv', index=False)
#4.2
columns = ["date", "place_id", "country_name",  "new_confirmed", "cumulative_confirmed"]
sickenings_df = df[columns]
sickenings_df = sickenings_df.loc[~sickenings_df.duplicated(subset=['date', 'place_id'], keep='last')]
sickenings_df.to_csv('0402_sickenings.csv', index=False)
#4.3
columns = ["date", "place_id", "country_name", "new_deceased", "cumulative_deceased"]
deceases_df = df[columns]
deceases_df = deceases_df.loc[~deceases_df.duplicated(subset=['date', 'place_id'], keep='last')]
deceases_df.to_csv('0403_deceases.csv', index=False)
#4.4
columns = ["date", "place_id", "country_name", "new_persons_vaccinated",                                            \
           "cumulative_persons_vaccinated", "new_persons_fully_vaccinated", "cumulative_persons_fully_vaccinated",  \
           "new_vaccine_doses_administered", "cumulative_vaccine_doses_administered"]
vaccinations_df = df[columns]
vaccinations_df = vaccinations_df.loc[~vaccinations_df.duplicated(subset=['date', 'place_id'], keep='last')]
vaccinations_df.to_csv('0404_vaccinations.csv', index=False)
#4.5 - Trendy zależne od wyleczonych pacjentów
columns = ["date", "place_id", "country_name", "new_recovered"]
recoveries_df = df[columns]
recoveries_df = recoveries_df.loc[~recoveries_df.duplicated(subset=['date', 'place_id'], keep='last')]
recoveries_df.to_csv('0405_recoveries.csv', index=False)

#5 ----------------------------------------------------------------------------

combined_df = pd.merge(vaccinations_df, recoveries_df, on=['date', 'place_id', "country_name"], how='outer')
combined_df = pd.merge(sickenings_df, combined_df, on=['date', 'place_id', "country_name"], how='outer')
combined_df = pd.merge(deceases_df, combined_df, on=['date', 'place_id', 'country_name'], how='outer')
combined_df = pd.merge(countries_df, combined_df, on=['country_name'], how='right')
combined_df.to_csv('0501_combined.csv', index=False)

#6 ----------------------------------------------------------------------------
temp_country_df1 = pd.read_csv('0401_countries.csv')
country_names_df1 = set(temp_country_df1['country_name'].dropna())
temp_country_df2 = pd.read_csv('world_countries.csv')
temp_country_df3 = pd.read_csv('gdp.csv')

# country_names_df2 = set(temp_country_df2['Country/Territory'].dropna())
# unique_df1 = sorted(country_names_df1 - country_names_df2)
# unique_df2 = sorted(country_names_df2 - country_names_df1)
# print("Kraje tylko w pliku 1:", unique_df1)
# print("Kraje tylko w pliku 2:", unique_df2)
#6.1
name_converter = {
    'Curacao': 'Curaçao',
    'DR Congo': 'Democratic Republic of the Congo',
    'Eswatini': 'Swaziland',
    'North Macedonia': 'Macedonia',
    'Reunion': 'Réunion',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Timor-Leste': 'East Timor',
    'United States': 'United States of America'
}
temp_country_df2['Country/Territory'] = temp_country_df2['Country/Territory'].apply(lambda x: name_converter.get(x, x))
temp_country_df2 = temp_country_df2.rename(columns={'Country/Territory': 'country_name'})
combined_df = pd.merge(combined_df, temp_country_df2, on='country_name', how='outer')
#
# country_names_df3 = set(temp_country_df3['Country Name'].dropna())
# unique_df1b = sorted(country_names_df1 - country_names_df3)
# unique_df3 = sorted(country_names_df3 - country_names_df1)
# print("Kraje tylko w pliku 1:", unique_df1b)
# print("Kraje tylko w pliku 2:", unique_df3)
#
#OR, ale wymagana zmiana słowniczka
# short_df2 = set(temp_country_df2['CCA3'].dropna())
# short_df3 = set(temp_country_df3['Country Code'].dropna())
# uniqueSh_df2 = sorted(short_df2 - short_df3)
# uniqueSh_df3 = sorted(short_df3 - short_df2)
# print("Skróty tylko w pliku 2:", uniqueSh_df2)
# print("Skróty tylko w pliku 3:", uniqueSh_df3)
#6.2
name_converter = {
    'Bahamas, The': 'Bahamas',
    'Brunei Darussalam': 'Brunei',
    'Cabo Verde': 'Cape Verde',
    'Caribbean small states': 'Caribbean Netherlands',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Congo, Rep.': 'Republic of the Congo',
    "Cote d'Ivoire": 'Ivory Coast',
    'Egypt, Arab Rep.': 'Egypt',
    'Gambia, The': 'Gambia',
    'Hong Kong SAR, China': 'Hong Kong',
    'Iran, Islamic Rep.': 'Iran',
    'Korea, Rep.': 'South Korea',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'Macedonia, FYR': 'Macedonia',
    'Micronesia, Fed. Sts.': 'Micronesia',
    'Netherlands Antilles': 'Netherlands Antilles',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Slovak Republic': 'Slovakia',
    'Syrian Arab Republic': 'Syria',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Venezuela, RB': 'Venezuela',
    'Yemen, Rep.': 'Yemen',
    'Timor-Leste': 'East Timor'
}
temp_country_df3['Country Name'] = temp_country_df3['Country Name'].apply(lambda x: name_converter.get(x, x))
temp_country_df3 = temp_country_df3.rename(columns={'Country Name': 'country_name'})
temp_country_df3 = temp_country_df3.pivot(index=['country_name', 'Country Code'], columns='Year', values='Value').reset_index()
temp_country_df3.rename(columns=lambda x: f"gdp_{x}" if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else x, inplace=True)

combined_df = combined_df.rename(columns={'date_y': 'date'})
combined_df = pd.merge(combined_df, temp_country_df3, on=['country_name'], how='outer')
combined_df.to_csv('0601_combined.csv', index=False)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Zad 2

#2.4.1
combined_df['sickenings_per_1000'] = combined_df['cumulative_confirmed'] / combined_df['population'] * 1000

# last_country_entry = combined_df
# last_country_entry = last_country_entry.dropna(subset=['sickenings_per_1000'])
# last_country_entry = last_country_entry.groupby('country_name')['sickenings_per_1000'].max()
# last_country_entry = last_country_entry.reset_index()
# print(last_country_entry[['country_name', 'sickenings_per_1000']].
#         sort_values(by='sickenings_per_1000', ascending=False).head(10))

combined_df = combined_df[combined_df['sickenings_per_1000'] <= 40000]

ax= sns.lineplot(data=combined_df, x='date', y='sickenings_per_1000', hue='country_name', marker='o')
plt.title('Sickenings per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sickenings per 1000 pupulation', fontsize=12)

plt.ylim(0, combined_df['sickenings_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)

ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

last_country_entry = combined_df
last_country_entry = last_country_entry.dropna(subset=['sickenings_per_1000'])
last_country_entry = last_country_entry.groupby('country_name')['date'].max()
last_country_entry = pd.merge(combined_df, last_country_entry, on=['country_name', 'date'], how='right')
print()
print("Top 10 countries with sickenings per 1000")
print(last_country_entry[['country_name', 'date', 'sickenings_per_1000']].
        sort_values(by='sickenings_per_1000', ascending=False).head(10))

#2.4.2
combined_df['vaccinations_per_1000'] = combined_df['cumulative_vaccine_doses_administered'] / combined_df['population'] * 1000

combined_df = combined_df[combined_df['vaccinations_per_1000'] <= 400000]

ax= sns.lineplot(data=combined_df, x='date', y='vaccinations_per_1000', hue='country_name', marker='o')
plt.title('Vaccinations per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Vaccinations per 1000 pupulation', fontsize=12)

plt.ylim(0, combined_df['vaccinations_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)

ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

last_country_entry = combined_df
last_country_entry = last_country_entry.dropna(subset=['vaccinations_per_1000'])
last_country_entry = last_country_entry.groupby('country_name')['date'].max()
last_country_entry = pd.merge(combined_df, last_country_entry, on=['country_name', 'date'], how='right')
print()
print("Top 10 countries with vaccinations per 1000")
print(last_country_entry[['country_name', 'date', 'vaccinations_per_1000']].
        sort_values(by='vaccinations_per_1000', ascending=False).head(10))

#2.4.3
combined_df['deaths_per_1000'] = combined_df['cumulative_deceased'] / combined_df['population'] * 1000

combined_df = combined_df[combined_df['deaths_per_1000'] <= 4000]

ax= sns.lineplot(data=combined_df, x='date', y='deaths_per_1000', hue='country_name', marker='o')
plt.title('Deaths per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Deaths per 1000 pupulation', fontsize=12)

plt.ylim(0, combined_df['deaths_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)

ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

last_country_entry = combined_df
last_country_entry = last_country_entry.dropna(subset=['deaths_per_1000'])
last_country_entry = last_country_entry.groupby('country_name')['date'].max()
last_country_entry = pd.merge(combined_df, last_country_entry, on=['country_name', 'date'], how='right')
print()
print("Top 10 countries with deaths per 1000")
print(last_country_entry[['country_name', 'date', 'deaths_per_1000']].
        sort_values(by='deaths_per_1000', ascending=False).head(10))

#2.4.4 (Grupy wiekowe 0-9, 10-19, 20-29, 30-39)
combined_df['ages_00-09_per_1000'] = combined_df['population_age_00_09'] / combined_df['population'] * 1000
combined_df['ages_10-19_per_1000'] = combined_df['population_age_10_19'] / combined_df['population'] * 1000
combined_df['ages_20-29_per_1000'] = combined_df['population_age_20_29'] / combined_df['population'] * 1000
combined_df['ages_30-39_per_1000'] = combined_df['population_age_30_39'] / combined_df['population'] * 1000

combined_df = combined_df[combined_df['ages_00-09_per_1000'] <= 500]
combined_df = combined_df[combined_df['ages_10-19_per_1000'] <= 500]
combined_df = combined_df[combined_df['ages_20-29_per_1000'] <= 500]
combined_df = combined_df[combined_df['ages_30-39_per_1000'] <= 500]

ax= sns.lineplot(data=combined_df, x='date', y='ages_00-09_per_1000', hue='country_name', marker='o')
plt.title('Ages 00-09 per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Ages 00-09 per 1000 pupulation', fontsize=12)
plt.ylim(0, combined_df['ages_00-09_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)
ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

ax= sns.lineplot(data=combined_df, x='date', y='ages_10-19_per_1000', hue='country_name', marker='o')
plt.title('Ages 10-19 per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Ages 10-19 per 1000 pupulation', fontsize=12)
plt.ylim(0, combined_df['ages_10-19_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)
ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

ax= sns.lineplot(data=combined_df, x='date', y='ages_20-29_per_1000', hue='country_name', marker='o')
plt.title('Ages 20-29 per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Ages 20-29 per 1000 pupulation', fontsize=12)
plt.ylim(0, combined_df['ages_20-29_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)
ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

ax= sns.lineplot(data=combined_df, x='date', y='ages_30-39_per_1000', hue='country_name', marker='o')
plt.title('Ages 30-39 per 1000 population in time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Ages 30-39 per 1000 pupulation', fontsize=12)
plt.ylim(0, combined_df['ages_30-39_per_1000'].max()*1.1)
plt.xlim(combined_df['date'].min() - pd.DateOffset(months=1), combined_df['date'].max() + pd.DateOffset(months=1))
plt.xticks(rotation=20)
ax.get_legend().remove()
plt.tight_layout()
plt.grid(True)
plt.show()

last_country_entry = combined_df
last_country_entry1 = last_country_entry; last_country_entry2 = last_country_entry;
last_country_entry3 = last_country_entry; last_country_entry4 = last_country_entry;
last_country_entry1 = last_country_entry.dropna(subset=['ages_00-09_per_1000'])
last_country_entry2 = last_country_entry.dropna(subset=['ages_10-19_per_1000'])
last_country_entry3 = last_country_entry.dropna(subset=['ages_20-29_per_1000'])
last_country_entry4 = last_country_entry.dropna(subset=['ages_30-39_per_1000'])

last_country_entry1 = last_country_entry1.groupby('country_name')['date'].max()
last_country_entry1 = pd.merge(combined_df, last_country_entry1, on=['country_name', 'date'], how='right')
print()
print("Top 10 population ages 0-9 per 1000")
print(last_country_entry1[['country_name', 'date', 'ages_00-09_per_1000']].
        sort_values(by='ages_00-09_per_1000', ascending=False).head(10))

last_country_entry2 = last_country_entry2.groupby('country_name')['date'].max()
last_country_entry2 = pd.merge(combined_df, last_country_entry2, on=['country_name', 'date'], how='right')
print()
print("Top 10 population ages 10-19 per 1000")
print(last_country_entry2[['country_name', 'date', 'ages_10-19_per_1000']].
        sort_values(by='ages_10-19_per_1000', ascending=False).head(10))

last_country_entry3 = last_country_entry3.groupby('country_name')['date'].max()
last_country_entry3 = pd.merge(combined_df, last_country_entry3, on=['country_name', 'date'], how='right')
print()
print("Top 10 population ages 20-29 per 1000")
print(last_country_entry3[['country_name', 'date', 'ages_20-29_per_1000']].
        sort_values(by='ages_20-29_per_1000', ascending=False).head(10))

last_country_entry4 = last_country_entry.groupby('country_name')['date'].max()
last_country_entry4 = pd.merge(combined_df, last_country_entry4, on=['country_name', 'date'], how='right')
print()
print("Top 10 population ages 30-39 per 1000")
print(last_country_entry4[['country_name', 'date', 'ages_30-39_per_1000']].
        sort_values(by='ages_30-39_per_1000', ascending=False).head(10))