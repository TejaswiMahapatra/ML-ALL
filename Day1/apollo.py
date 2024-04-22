import csv

with open('Dataset_Day1.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    cities_with_apollo_hospital = set()

    for row in reader:
        city = row[0]
        hospital = row[1]
        if "apollo" in hospital.lower():
            cities_with_apollo_hospital.add(city)

print(cities_with_apollo_hospital)
