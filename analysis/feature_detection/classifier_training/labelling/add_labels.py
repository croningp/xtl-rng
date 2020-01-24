import os
import sys
import csv

original_file = 'via_region_data.csv'
new_file = 'via_region_data2.csv'

def load_csv(original_file):
    data = []
    with open(original_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def annotation_data(data):
    last_cell = '{"object_name":"crystal","object_color":"blue"}'
    for idx, row in enumerate(data):
        if idx > 0:
            data[idx][-1] = last_cell
    return data

def save_csv(data, new_file):

    with open(new_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

def remove_duplicates(data):
    crystals = []
    for row in data:
        pass




data = load_csv(original_file)
data = annotation_data(data)

save_csv(data, new_file)
