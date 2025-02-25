# analyze_sensitivity.py

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_sensitivities(input_file):
    with open(input_file, "r") as f:
        return json.load(f)

def analyze_and_plot(sensitivities):
    # Flatten sensitivities into a list of (layer_name, sensitivity) pairs
    data = []
    for layer_name, values in sensitivities.items():
        avg_sensitivity = np.mean(values)
        data.append((layer_name, avg_sensitivity))

    # Cluster sensitivities using KMeans
    sensitivities_array = np.array([x[1] for x in data]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(sensitivities_array)
    labels = kmeans.labels_

    # Plot results
    plt.figure(figsize=(10, 6))
    for i, (layer_name, sensitivity) in enumerate(data):
        plt.scatter(i, sensitivity, c=labels[i], cmap="viridis", label=layer_name if i < 10 else "")
    plt.xlabel("Layer Index")
    plt.ylabel("Average Sensitivity")
    plt.title("Layer Sensitivities Clustered by KMeans")
    plt.colorbar(label="Cluster")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with sensitivities")
    args = parser.parse_args()

    sensitivities = load_sensitivities(args.input_file)
    analyze_and_plot(sensitivities)

if __name__ == "__main__":
    main()