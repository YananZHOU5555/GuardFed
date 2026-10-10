def score(row):
    gap = max(row['aeod'], row['aspd'])
    return row['accuracy'] - .35 * (.45 * row['aeod'] + .45 * row['aspd'] + .10 * gap) - .10 * max(0, gap - .06)
