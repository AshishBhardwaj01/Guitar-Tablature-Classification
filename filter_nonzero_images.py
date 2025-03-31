import os
import numpy as np
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def filter_nonzero_tablatures(tablature_segments_dir, cqt_images_dir, output_dir):
    """
    Filter tablature files that have at least one non-zero value (indicating a played string)
    and copy their corresponding PNG images to the output directory.
    
    Args:
        tablature_segments_dir (str): Directory containing the tablature .npy files
        cqt_images_dir (str): Directory containing the CQT PNG images
        output_dir (str): Directory to save the filtered images and tablatures
    """
    # Create output directories
    filtered_images_dir = os.path.join(output_dir, 'filtered_images')
    filtered_tablatures_dir = os.path.join(output_dir, 'filtered_tablatures')
    
    os.makedirs(filtered_images_dir, exist_ok=True)
    os.makedirs(filtered_tablatures_dir, exist_ok=True)
    
    # Get list of all tablature files
    tab_files = [f for f in os.listdir(tablature_segments_dir) if f.endswith('.npy')]
    
    # Counter for statistics
    total_files = len(tab_files)
    filtered_count = 0
    skipped_count = 0
    missing_image_count = 0
    
    print(f"Processing {total_files} tablature files...")
    
    for tab_file in tqdm(tab_files):
        # Load tablature file
        tab_path = os.path.join(tablature_segments_dir, tab_file)
        try:
            tablature = np.load(tab_path)
            
            # Check if the tablature has at least one non-zero value
            if np.any(tablature > 0):
                # Find corresponding image file
                base_name = os.path.splitext(tab_file)[0]
                img_file = f"{base_name}.png"
                img_path = os.path.join(cqt_images_dir, img_file)
                
                if os.path.exists(img_path):
                    # Copy both files to output directory
                    shutil.copy2(img_path, os.path.join(filtered_images_dir, img_file))
                    shutil.copy2(tab_path, os.path.join(filtered_tablatures_dir, tab_file))
                    filtered_count += 1
                else:
                    missing_image_count += 1
                    print(f"Warning: Missing image file for {tab_file}")
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing {tab_file}: {e}")
    
    # Print statistics
    print(f"\nFilter complete:")
    print(f"Total tablature files: {total_files}")
    print(f"Files with non-zero values: {filtered_count}")
    print(f"Files with all zeros (skipped): {skipped_count}")
    print(f"Files missing corresponding image: {missing_image_count}")
    
    return filtered_images_dir, filtered_tablatures_dir

def visualize_examples(filtered_images_dir, filtered_tablatures_dir, num_examples=5):
    """
    Visualize a few examples of the filtered data
    """
    # Get paired files
    tab_files = sorted([f for f in os.listdir(filtered_tablatures_dir) if f.endswith('.npy')])
    
    # Select random examples
    if len(tab_files) < num_examples:
        num_examples = len(tab_files)
    
    sample_indices = np.random.choice(len(tab_files), num_examples, replace=False)
    
    # Plot examples
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3*num_examples))
    
    for i, idx in enumerate(sample_indices):
        tab_file = tab_files[idx]
        base_name = os.path.splitext(tab_file)[0]
        img_file = f"{base_name}.png"
        
        # Load tablature
        tab_path = os.path.join(filtered_tablatures_dir, tab_file)
        tablature = np.load(tab_path)
        
        # Load image
        img_path = os.path.join(filtered_images_dir, img_file)
        image = plt.imread(img_path)
        
        # Plot
        if num_examples == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[i]
            
        ax1.imshow(image)
        ax1.set_title(f"CQT Image: {img_file}")
        ax1.axis('off')
        
        # Create tablature visualization
        ax2.imshow(tablature, cmap='viridis', interpolation='nearest')
        ax2.set_title(f"Tablature: {tab_file}")
        ax2.set_ylabel("String")
        ax2.set_xlabel("Fret")
        for string in range(tablature.shape[0]):
            for fret in range(tablature.shape[1]):
                if tablature[string, fret] > 0:
                    ax2.text(fret, string, str(int(tablature[string, fret])), 
                             ha="center", va="center", color="white")
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(filtered_images_dir), 'examples.png'))
    plt.close()
    
    print(f"Example visualization saved to {os.path.join(os.path.dirname(filtered_images_dir), 'examples.png')}")

if __name__ == "__main__":
    # Set your directories here
    tablature_segments_dir = "path/to/tablature_segments"
    cqt_images_dir = "path/to/cqt_images"
    output_dir = "path/to/filtered_data"
    
    # Filter the data
    filtered_images_dir, filtered_tablatures_dir = filter_nonzero_tablatures(
        tablature_segments_dir, cqt_images_dir, output_dir
    )
    
    # Visualize some examples
    visualize_examples(filtered_images_dir, filtered_tablatures_dir)
    
    print(f"\nFiltered images saved to: {filtered_images_dir}")
    print(f"Filtered tablatures saved to: {filtered_tablatures_dir}")
    
    # Print instructions for using with your data loader
    print("\nTo use with your existing data loader, update your paths:")
    print("train_loader, val_loader, test_loader = create_dataloaders(")
    print(f"    audio_dir='{filtered_images_dir}',")
    print(f"    annotation_dir='{filtered_tablatures_dir}',")
    print("    batch_size=64")
    print(")")
