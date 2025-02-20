import numpy as np
import matplotlib.pyplot as plt

def read_npz_file(npz_file):
    # Load the .npz file with allow_pickle=True
    data = np.load(npz_file, allow_pickle=True)
    
    # Print the keys (names of arrays stored in the .npz file)
    print("Keys in the .npz file:")
    for key in data.keys():
        print(key)
        
    # Print the arrays corresponding to each key
    print("\nContents of the .npz file:")
    for key in data.keys():
        print(f"\n{key}:\n", data[key])

    return data

def visualize_distance_map(distance_matrix):
    """
    Visualizes the distance map using Matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(label="Distance (Å)")
    plt.title("Protein Distance Map")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.show()

if __name__ == "__main__":
    # Specify the .npz file path
    npz_file_path = "C:/Users/LENOVO/Desktop/protein-go-predictor/AF-Q6K7V6-F1-model_v4_A.npz"
    
    # Read the contents of the .npz file
    npz_data = read_npz_file(npz_file_path)

    # Extract and visualize the distance matrix
    if 'C_alpha' in npz_data:
        distance_matrix = npz_data['C_alpha']
        visualize_distance_map(distance_matrix)
    else:
        print("Error: 'C_alpha' key not found in the .npz file.")
