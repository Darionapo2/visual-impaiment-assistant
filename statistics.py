import pandas as pd

# Funzione per calcolare i parametri statistici
def calcola_statistiche(data, label):
    filtered_data = data[data['Label'] == label]['Time']

    media = filtered_data.mean()
    varianza = filtered_data.var()
    deviazione_standard = filtered_data.std()
    minimo = filtered_data.min()
    massimo = filtered_data.max()

    return {
        'Average': media,
        'Variance': varianza,
        'Standard Deviation': deviazione_standard,
        'Min': minimo,
        'Max': massimo
    }


# Lettura del file CSV
# Assicurati che il file CSV abbia colonne "Label" e "Time"
data = pd.read_csv('detection_times.csv')

# Calcolo delle statistiche per OBSTACLE DETECTION
stat_obstacle = calcola_statistiche(data, 'OBSTACLE DETECTION')

# Calcolo delle statistiche per OBJECT DETECTION
stat_object = calcola_statistiche(data, 'OBJECT DETECTION')

# Stampa dei risultati
print("Statistics for OBSTACLE DETECTION:")
for key, value in stat_obstacle.items():
    print(f"{key}: {value}")

print("\nStatistics for OBJECT DETECTION:")
for key, value in stat_object.items():
    print(f"{key}: {value}")
