import h5py

def print_weight_shapes(weight_file):
    """
    Print the shapes of all weights in an H5 model weights file
    """
    print(f"\nExamining weights in: {weight_file}\n" + "-"*50)
    
    # Open the H5 file and explore its structure
    with h5py.File(weight_file, 'r') as f:
        # Print all top-level keys (layer names)
        print("Layers in the model:")
        for key in f.keys():
            print(f"  {key}")
        
        print("\nDetailed weight shapes:")
        # Iterate through all layers and their weights
        for layer_name in f.keys():
            layer = f[layer_name]
            # Each layer has a 'weight_names' attribute that lists all weight names
            if 'weight_names' in layer.attrs:
                weight_names = [w.decode('utf8') for w in layer.attrs['weight_names']]
                for weight_name in weight_names:
                    weight = layer[weight_name]
                    print(f"  {weight_name}: {weight.shape}")