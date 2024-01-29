import os
import pandas as pd
from tqdm import tqdm

def compare_files(main_file, other_file):
    # Lese beide Dateien und konvertiere sie in Sets von Tuples
    main_df = pd.read_csv(main_file)
    other_df = pd.read_csv(other_file)

    main_set = set(map(tuple, main_df.values))
    other_set = set(map(tuple, other_df.values))

    # Finde die Schnittmenge der Sets
    matching_rows = len(main_set.intersection(other_set))

    # Berechne das Rating
    rating = matching_rows / len(main_df)
    return rating

def main():
    main_file = 'submission.csv'
    submissions_folder = 'data/submissions'
    ratings = []

    # Erstelle eine Liste aller CSV-Dateien im Ordner
    csv_files = [f for f in os.listdir(submissions_folder) if f.endswith('.csv')]

    # Durchlaufe alle Dateien im Ordner mit einer Statusleiste
    for filename in tqdm(csv_files, desc="Vergleiche Dateien"):
        other_file = os.path.join(submissions_folder, filename)
        rating = compare_files(main_file, other_file)
        ratings.append((filename, rating))

    # Sortiere die Ergebnisse nach dem Rating
    sorted_ratings = sorted(ratings, key=lambda x: x[1], reverse=True)

    # Drucke die sortierten Ergebnisse
    for filename, rating in sorted_ratings:
        print(f'Ähnlichkeitsrating für {filename}: {rating:.4f}')

if __name__ == '__main__':
    main()