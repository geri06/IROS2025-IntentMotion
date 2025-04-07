import csv

# Read the CSV file
with open('times.csv', 'r') as file:
    values = [float(line.strip()) for line in file]

# Compute the result
result = round(sum(values) / 100, 3)

# Print the result
print(result)