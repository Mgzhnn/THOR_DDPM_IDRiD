import csv


def get_data_from_csv(path_to_csv):
    """
    :param path_to_csv: str
        path to csv with filenames
    :return: list
        list with all the filenames
    """
    files = []
    if type(path_to_csv) is not list:
        path_to_csv = [path_to_csv]
    for single_csv in path_to_csv:
        ct = 0
        with open(single_csv, newline='') as csv_file:
            p_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in p_reader:
                if type(row) == list:
                    row = row[0]
                ct += 1
                if ct == 1:
                    continue
                files.append(row)
    return files
