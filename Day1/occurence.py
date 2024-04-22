import csv

with open('Dataset_Day1.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skips the first row
    city_count = {}
    for row in reader:
        city = row[0]
        if city in city_count:
            city_count[city] += 1
        else:
            city_count[city] = 1
print(city_count)
