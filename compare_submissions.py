import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, fbeta_score

def calculate_metrics(main_set, other_set, y_true, y_pred):
    """
    Berechnet die ursprüngliche Metrik und den F-beta-Score.
    """
    # Alte Metrik: Verhältnis der übereinstimmenden Zeilen zur Gesamtzahl der Zeilen in main_set
    matching_rows = len(main_set.intersection(other_set))
    old_metric = matching_rows / len(main_set)

    # F-beta-Score
    fbeta = fbeta_score(y_true, y_pred, beta=1.75)

    return old_metric, fbeta

def compare_files(main_file, other_file):
    """
    Vergleicht zwei Dateien und berechnet die Metriken.
    """
    # Lese beide Dateien
    main_df = pd.read_csv(main_file)
    other_df = pd.read_csv(other_file)

    # Konvertiere die Daten in Sets von Tuples für die alte Metrik
    main_set = set(map(tuple, main_df.values))
    other_set = set(map(tuple, other_df.values))

    # Annahme: Die erste Spalte ist die ID, die zweite Spalte ist das Label
    y_true = main_df.iloc[:, 1]
    y_pred = other_df.iloc[:, 1]

    return calculate_metrics(main_set, other_set, y_true, y_pred)

def main():
    main_file = 'submission.csv'
    submissions_folder = 'data/submissions'
    ratings = []

    # Erstelle eine Liste aller CSV-Dateien im Ordner
    csv_files = [f for f in os.listdir(submissions_folder) if f.endswith('.csv')]

    # Durchlaufe alle Dateien und bewerte sie
    for filename in tqdm(csv_files, desc="Vergleiche Dateien"):
        other_file = os.path.join(submissions_folder, filename)
        old_metric, fbeta = compare_files(main_file, other_file)
        ratings.append((filename, old_metric, fbeta))

    # Sortiere die Ergebnisse
    sorted_ratings = sorted(ratings, key=lambda x: (x[1], x[2]))

    # Drucke die sortierten Ergebnisse
    for filename, old_metric, fbeta in sorted_ratings:
        print(f'{filename}: Alter Score: {old_metric:.4f}, F-beta-Score: {fbeta:.4f}')

if __name__ == '__main__':
    main()
