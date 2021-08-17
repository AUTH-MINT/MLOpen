import os
import csv
import pandas as pd


def prepare_data(paths, sentiments):
    """
    Import data from files. A set of directory paths pointing to directories contaning statement
    files, and a second set of corresponding sentiments are used.
    """
    df = read_from_dir(paths[0], sentiments[0])
    for path, sentiment in zip(paths[1:], sentiments[1:]):
        df1 = read_from_dir(path, sentiment)
        df = pd.concat([df, df1], ignore_index=True)
        # df3 = pd.DataFrame([['This is just a movie', 2], ['this is a movie', 2], ['i went to a movie', 2], ['going to the movies', 2]],  columns=['text', 'sentiment'])
        # df = pd.concat([df, df3], ignore_index=True)
    return df


def read_from_files(text, sentiment):
    """
    Read all lines of a file
    """
    data = []
    cnt = 0
    with open(text, "r") as lines:
        for line in lines.get_lines():
            data.append([text, sentiment])
            cnt += 1
            if cnt > 1000:
                break
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    return df


def read_from_dir(dir, sentiment=None):
    """
    Read all files of a directory as statements
    """
    data = []
    cnt = 0
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as f:
            text = f.read()
            if sentiment is not None:
                data.append([text, sentiment])
            else:
                data.append(text)
            cnt += 1
            if cnt > 1000:
                break
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    return df


def read_multiple_to_csv(dir, sentiment=None):
    """
    Write to csv
    """
    _csv = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), 'r') as f:
            text = f.read()
            if sentiment is not None:
                _csv.append([text, sentiment])
            else:
                _csv.append(text)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                           os.pardir, 'data', 'csvs', dir.split(os.path.sep)[-2
                                                      ] + '.csv'), 'w') as out:
        csv_out = csv.writer(out)
        for row in _csv:
            csv_out.writerow(row)


def get_from_csvs(dir=None, csvs=None):
    """
    Import from csv
    """
    if dir is not None:
        df = None
        for filename in os.listdir(dir):
            df = pd.read_csv(os.path.join(dir, filename)) if df is None else pd.concat(
                [df, pd.read_csv(os.path.join(dir, filename))], ignore_index=True)
        return df
    elif type(csvs) is list:
        df = pd.read_csv(csvs[0])
        for cs in csvs[1:]:
            df = pd.concat([df, pd.read_csv(cs)], ignore_index=True)
        return df
