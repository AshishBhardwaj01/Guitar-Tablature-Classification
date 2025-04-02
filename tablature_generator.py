import os
import re
import numpy as np
import jams
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path

class TablatureGenerator:
    def __init__(self, jams_folder, cqt_folder, output_folder):
        """
        Initialize the tablature generator.
        
        Args:
            jams_folder (str): Path to folder containing JAMS annotation files
            cqt_folder (str): Path to folder containing CQT image files
            output_folder (str): Path where tablature files will be saved
        """
        self.jams_folder = jams_folder
        self.cqt_folder = cqt_folder
        self.output_folder = output_folder
        self.n_strings = 6
        self.n_frets = 19  # Including open position (0th fret)
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Guitar string tuning (standard): E2, A2, D3, G3, B3, E4
        # MIDI note numbers: 40, 45, 50, 55, 59, 64
        self.string_tunings = [40, 45, 50, 55, 59, 64]
        
    def get_base_name_from_cqt(self, cqt_filename):
        """Extract base name from CQT image filename."""
        # Example: 05_Funk2-119-G_comp_segment_318_7.60.png
        # Base name: 05_Funk2-119-G_comp
        match = re.match(r'(.+?)_segment_', os.path.basename(cqt_filename))
        if match:
            return match.group(1)
        return None
        
    def get_time_from_cqt(self, cqt_filename):
        """Extract time in seconds from CQT image filename."""
        # Example: 05_Funk2-119-G_comp_segment_318_7.60.png
        # Time: 7.60
        match = re.search(r'_(\d+\.\d+)\.png$', os.path.basename(cqt_filename))
        if match:
            return float(match.group(1))
        return None
    
    def find_matching_jams(self, base_name):
        """Find JAMS file matching the base name of a CQT image."""
        jams_files = glob(os.path.join(self.jams_folder, f"{base_name}*.jams"))
        if jams_files:
            return jams_files[0]
        return None
    
    def note_to_tab_position(self, midi_note):
        """
        Convert MIDI note to string and fret position on guitar.
        
        Args:
            midi_note (int): MIDI note number
            
        Returns:
            tuple: (string_idx, fret) or None if note can't be played
        """
        for string_idx, open_note in enumerate(self.string_tunings):
            # Check if the note can be played on this string
            if midi_note >= open_note and midi_note <= open_note + self.n_frets - 1:
                fret = midi_note - open_note
                return string_idx, fret
        
        # If the note is outside the range of the guitar
        return None
    
    def create_tablature_from_notes(self, notes, start_time, end_time):
        """
        Create tablature matrix for a time segment based on MIDI notes.
        
        Args:
            notes: List of (time, duration, midi_note) tuples
            start_time (float): Start time of the segment in seconds
            end_time (float): End time of the segment in seconds
            
        Returns:
            numpy.ndarray: Tablature matrix (n_strings x n_frets)
        """
        # Initialize tablature matrix with zeros
        # First column is for open strings (0th fret)
        tab = np.zeros((self.n_strings, self.n_frets), dtype=np.int8)
        
        # Track which strings are used
        used_strings = set()
        
        # Find notes that overlap with the time segment
        for time, duration, midi_note in notes:
            note_end_time = time + duration
            
            # Check if the note overlaps with the segment
            if not (note_end_time <= start_time or time >= end_time):
                position = self.note_to_tab_position(midi_note)
                if position:
                    string_idx, fret = position
                    tab[string_idx, fret] = 1
                    used_strings.add(string_idx)
        
        # Mark open strings (first column) as 1 if the string is not used
        for string_idx in range(self.n_strings):
            if string_idx not in used_strings:
                tab[string_idx, 0] = 1
                
        return tab
    
    def extract_notes_from_jams(self, jams_file):
        """
        Extract notes from JAMS file.
        
        Args:
            jams_file (str): Path to JAMS file
            
        Returns:
            list: List of (time, duration, midi_note) tuples
        """
        try:
            jam = jams.load(jams_file)
            notes = []
            
            # Find note_midi annotations
            for ann in jam.annotations:
                if ann.namespace == 'note_midi':
                    for note in ann:
                        time = note.time
                        duration = note.duration
                        midi_note = int(round(note.value))
                        notes.append((time, duration, midi_note))
            
            return notes
        except Exception as e:
            print(f"Error loading JAMS file {jams_file}: {e}")
            return []
    
    def process_cqt_images(self):
        """
        Process all CQT images and create corresponding tablature files.
        """
        # Get all CQT image files
        cqt_files = glob(os.path.join(self.cqt_folder, '*.png'))
        print(f"Found {len(cqt_files)} CQT image files")
        
        # Track processed files for each base name
        processed_count = {}
        
        # Dictionary to store JAMS data to avoid reloading
        jams_data_cache = {}
        
        for cqt_file in cqt_files:
            base_name = self.get_base_name_from_cqt(cqt_file)
            if not base_name:
                print(f"Could not extract base name from {cqt_file}")
                continue
                
            time_point = self.get_time_from_cqt(cqt_file)
            if time_point is None:
                print(f"Could not extract time from {cqt_file}")
                continue
            
            # Define the time segment (0.2 seconds per image)
            start_time = time_point
            end_time = start_time + 0.2
            
            # Find matching JAMS file
            if base_name in jams_data_cache:
                notes = jams_data_cache[base_name]
            else:
                jams_file = self.find_matching_jams(base_name)
                if not jams_file:
                    print(f"No matching JAMS file found for {base_name}")
                    continue
                
                notes = self.extract_notes_from_jams(jams_file)
                jams_data_cache[base_name] = notes
                
            # Create tablature for this segment
            tab = self.create_tablature_from_notes(notes, start_time, end_time)
            
            # Generate output filename
            cqt_basename = os.path.basename(cqt_file)
            output_filename = cqt_basename.replace('.png', '.npy')
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Save tablature as numpy array
            np.save(output_path, tab)
            
            # Update count
            processed_count[base_name] = processed_count.get(base_name, 0) + 1
            
            # Debugging: Check if tablature contains any notes
            if np.sum(tab[:, 1:]) == 0:
                if np.sum(tab) > 0:
                    print(f"Warning: {output_filename} only has open strings, no fretted notes")
                else:
                    print(f"Warning: {output_filename} is completely empty (all zeros)")
        
        # Print summary
        print("\nProcessing Summary:")
        for base, count in processed_count.items():
            print(f"{base}: {count} files processed")
        
        print(f"\nTotal files processed: {sum(processed_count.values())}")
        print(f"Tablature files saved to: {self.output_folder}")
    
    def visualize_tablature(self, tab_file):
        """
        Visualize a tablature file for verification.
        
        Args:
            tab_file (str): Path to tablature .npy file
        """
        tab = np.load(tab_file)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(tab, cmap='Blues', interpolation='none')
        plt.colorbar()
        plt.title(f"Tablature: {os.path.basename(tab_file)}")
        plt.xlabel("Fret Position (0 = open string)")
        plt.ylabel("String (0 = lowest E string)")
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(range(self.n_frets))
        plt.yticks(range(self.n_strings), ['E', 'A', 'D', 'G', 'B', 'e'])
        
        # Highlight played positions
        for i in range(self.n_strings):
            for j in range(self.n_frets):
                if tab[i, j] == 1:
                    plt.text(j, i, 'X', ha='center', va='center', color='red')
        
        plt.tight_layout()
        
        # Create visualizations directory
        vis_dir = os.path.join(self.output_folder, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save the visualization
        output_path = os.path.join(vis_dir, os.path.basename(tab_file).replace('.npy', '.png'))
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def create_html_index(self):
        """
        Create an HTML index file for downloading the tablature files.
        """
        index_path = os.path.join(self.output_folder, 'index.html')
        tab_files = glob(os.path.join(self.output_folder, '*_tab.npy'))
        
        # Get visualization samples
        sample_count = min(5, len(tab_files))
        sample_vis = []
        if sample_count > 0:
            samples = np.random.choice(tab_files, sample_count, replace=False)
            for sample in samples:
                vis_path = self.visualize_tablature(sample)
                rel_path = os.path.relpath(vis_path, self.output_folder)
                sample_vis.append(rel_path)
        
        with open(index_path, 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tablature Files</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    .file-list { margin-top: 20px; }
                    .file-item { margin: 5px 0; }
                    .file-link { text-decoration: none; color: #0066cc; }
                    .file-link:hover { text-decoration: underline; }
                    .sample-vis { margin-top: 30px; }
                    .sample-vis img { max-width: 100%; margin-bottom: 20px; border: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <h1>Tablature Files</h1>
                <p>Total files: ''' + str(len(tab_files)) + '''</p>
                <div class="file-list">
            ''')
            
            # Add file links
            for tab_file in sorted(tab_files):
                filename = os.path.basename(tab_file)
                f.write(f'<div class="file-item"><a href="{filename}" class="file-link" download>{filename}</a></div>\n')
            
            f.write('''
                </div>
                <div class="sample-vis">
                    <h2>Sample Visualizations</h2>
            ''')
            
            # Add sample visualizations
            for vis_path in sample_vis:
                f.write(f'<img src="{vis_path}" alt="Tablature Visualization">\n')
            
            f.write('''
                </div>
            </body>
            </html>
            ''')
        
        print(f"HTML index created at: {index_path}")
        return index_path

def main():
    """
    Main function to run the tablature generator.
    """
    # Get input paths from user
    print("Guitar Tablature Generator")
    print("=========================")
    
    jams_folder = "/content/drive/MyDrive/Seminar_8ThSEM_/Dataset/dataset_seminar_guitar_2025_/annotation"
    cqt_folder = "./cqt_images"
    output_folder = "./tablatures"
    
    # Create and run tablature generator
    generator = TablatureGenerator(jams_folder, cqt_folder, output_folder)
    
    print("\nProcessing CQT images and creating tablature files...")
    generator.process_cqt_images()
    
    # Create index for downloading
    print("\nCreating HTML index for downloading files...")
    index_path = generator.create_html_index()
    
    print("\nDone!")
    print(f"You can access the tablature files at: {index_path}")

if __name__ == "__main__":
    main()



# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as transforms
# import torchvision.models as models
# import argparse
# import warnings
# from tqdm import tqdm
# import soundfile as sf

# # Suppress warnings
# warnings.filterwarnings("ignore")

# class GuitarTabNet(nn.Module):
#     """
#     Recreation of the GuitarTabNet model architecture
#     """
#     def __init__(self, input_channels=3, num_frets=19):
#         super(GuitarTabNet, self).__init__()

#         # Load ResNet18 and modify first conv layer to accept RGB images
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.resnet.fc = nn.Linear(512, 256) 
        
#         # Separate fully connected layers for each string
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(256, 128),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(128),
#                 nn.Dropout(0.3),
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(64),
#                 nn.Dropout(0.2),
#                 nn.Linear(64, num_frets)
#             ) for _ in range(6)
#         ])

#     def forward(self, x):
#         # Feature extraction with ResNet
#         features = self.resnet(x)  
        
#         # Apply each string branch
#         outputs = [branch(features) for branch in self.branches]
#         return outputs

# class TablatureGenerator:
#     """
#     A class to handle the generation of guitar tablature from audio files
#     """
#     def __init__(self, model_path, device=None):
#         """
#         Initialize the tablature generator
        
#         Args:
#             model_path (str): Path to the .pt model file
#             device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to None.
#         """
#         # Set device
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)
        
#         print(f"Using device: {self.device}")
        
#         # Load model
#         self.model = self._load_model(model_path)
        
#         # Set up image transform
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),  # Resize to ResNet input size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
        
#         # Create temp directory for spectrograms
#         os.makedirs("temp_spectrograms", exist_ok=True)
    
#     def _load_model(self, model_path):
#         """
#         Load the model from file
        
#         Args:
#             model_path (str): Path to the .pt model file
            
#         Returns:
#             nn.Module: Loaded model
#         """
#         try:
#             model = GuitarTabNet(input_channels=3, num_frets=19).to(self.device)
#             checkpoint = torch.load(model_path, map_location=self.device)
            
#             # Check what's in the checkpoint
#             print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
#             # Handle both DataParallel and regular models
#             if 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             else:
#                 # Assume the checkpoint is directly a state dict
#                 state_dict = checkpoint
            
#             # Check if it's a DataParallel model
#             if list(state_dict.keys())[0].startswith('module.'):
#                 model = nn.DataParallel(model)
            
#             model.load_state_dict(state_dict)
#             print("Model loaded successfully")
#             return model
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise
    
#     def audio_to_cqt_image(self, audio_file, output_path=None, sr=22050, hop_length=512):
#         """
#         Convert audio file to CQT spectrogram image
        
#         Args:
#             audio_file (str): Path to audio file
#             output_path (str, optional): Path to save the spectrogram. Defaults to None.
#             sr (int, optional): Sample rate. Defaults to 22050.
#             hop_length (int, optional): Hop length. Defaults to 512.
            
#         Returns:
#             str: Path to saved spectrogram image
#         """
#         if output_path is None:
#             output_path = os.path.join("temp_spectrograms", f"{os.path.basename(audio_file)}_spectrogram.png")
        
#         # Load audio file
#         print(f"Loading audio file: {audio_file}")
#         y, sr = librosa.load(audio_file, sr=sr)
        
#         # Calculate CQT spectrogram
#         C = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C2'), 
#                       n_bins=84, bins_per_octave=12)
        
#         # Convert to magnitude and apply log transform
#         C_mag = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
#         # Create spectrogram image
#         plt.figure(figsize=(10, 5))
#         librosa.display.specshow(C_mag, sr=sr, x_axis='time', y_axis='cqt_note', 
#                                hop_length=hop_length)
#         plt.tight_layout()
        
#         # Save as image
#         plt.savefig(output_path)
#         plt.close()
        
#         print(f"Saved CQT spectrogram to {output_path}")
#         return output_path
    
#     def segment_audio(self, audio_file, segment_duration=3.0, sr=22050, overlap=0.5):
#         """
#         Split audio into segments for processing
        
#         Args:
#             audio_file (str): Path to audio file
#             segment_duration (float, optional): Duration of each segment in seconds. Defaults to 3.0.
#             sr (int, optional): Sample rate. Defaults to 22050.
#             overlap (float, optional): Overlap between segments (0-1). Defaults to 0.5.
            
#         Returns:
#             tuple: (List of segments, sample rate)
#         """
#         y, sr = librosa.load(audio_file, sr=sr)
#         segment_length = int(segment_duration * sr)
#         hop_length = int(segment_length * (1 - overlap))
#         segments = []
        
#         # Split audio into segments with overlap
#         for start in range(0, len(y), hop_length):
#             end = min(start + segment_length, len(y))
#             segment = y[start:end]
            
#             # If segment is too short, pad it
#             if len(segment) < segment_length:
#                 segment = np.pad(segment, (0, segment_length - len(segment)))
                
#             segments.append((segment, start / sr))  # Store segment and its start time
        
#         return segments, sr
    
#     def predict_tablature(self, image_path):
#         """
#         Run inference on spectrogram image to predict tablature
        
#         Args:
#             image_path (str): Path to spectrogram image
            
#         Returns:
#             list: Predicted fret positions for each string
#         """
#         # Load and preprocess image
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        
#         # Run inference
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(image_tensor)
            
#         # Convert outputs to tablature format
#         tablature = []
#         for output in outputs:
#             _, predicted_fret = torch.max(output, 1)
#             tablature.append(predicted_fret.item())
        
#         return tablature
    
#     def format_tablature(self, tablature_segments, timings=None):
#         """
#         Convert model predictions to readable guitar tablature format
        
#         Args:
#             tablature_segments (list): List of tablature segments
#             timings (list, optional): List of segment start times. Defaults to None.
            
#         Returns:
#             str: Formatted tablature notation
#         """
#         # Initialize strings for tablature
#         strings = [[] for _ in range(6)]  # 6 guitar strings
        
#         # Add time markers if available
#         time_markers = []
#         if timings:
#             for t in timings:
#                 time_markers.append(f"{t:.1f}s")
        
#         # Process each segment prediction
#         for segment in tablature_segments:
#             # Add predictions to each string (in reverse order for standard tablature notation)
#             for i, fret in enumerate(reversed(segment)):
#                 strings[i].append(str(fret))
        
#         # Format the tablature
#         tab_text = ""
#         string_names = ["e|", "B|", "G|", "D|", "A|", "E|"]  # Standard tuning
        
#         # Add time markers if available
#         if time_markers:
#             tab_text += "  " + "  ".join(time_markers) + "\n"
        
#         for i, string in enumerate(strings):
#             tab_line = string_names[i]
#             for fret in string:
#                 # Format each fret number with consistent spacing
#                 if len(fret) == 1:
#                     tab_line += f"{fret}--"
#                 else:
#                     tab_line += f"{fret}-"
#             tab_line += "|"
#             tab_text += tab_line + "\n"
        
#         return tab_text
    
#     def post_process_tablature(self, raw_tablature, smooth_window=3):
#         """
#         Apply post-processing to the raw tablature predictions
        
#         Args:
#             raw_tablature (list): List of raw tablature segments
#             smooth_window (int, optional): Window size for smoothing. Defaults to 3.
            
#         Returns:
#             list: Processed tablature
#         """
#         # If not enough segments for smoothing, return raw
#         if len(raw_tablature) <= smooth_window:
#             return raw_tablature
            
#         processed = []
        
#         # Convert to numpy for easier processing
#         tab_array = np.array(raw_tablature)
        
#         # Simple median filter for each string
#         for i in range(tab_array.shape[1]):  # For each string
#             string_data = tab_array[:, i]
            
#             # Apply median filter
#             for j in range(len(string_data)):
#                 start = max(0, j - smooth_window // 2)
#                 end = min(len(string_data), j + smooth_window // 2 + 1)
#                 window = string_data[start:end]
                
#                 # Only smooth if there's variance in the window
#                 if np.var(window) > 0:
#                     # Use mode (most common value) for smoothing discrete fret numbers
#                     values, counts = np.unique(window, return_counts=True)
#                     string_data[j] = values[np.argmax(counts)]
            
#             # Update the processed array
#             tab_array[:, i] = string_data
        
#         # Convert back to list format
#         for i in range(tab_array.shape[0]):
#             processed.append(tab_array[i].tolist())
        
#         return processed
    
#     def generate_tablature(self, audio_file, output_file=None, use_segments=True, 
#                            segment_duration=3.0, overlap=0.5, smooth=True):
#         """
#         Generate guitar tablature from an audio file
        
#         Args:
#             audio_file (str): Path to audio file
#             output_file (str, optional): Path to save the tablature. Defaults to None.
#             use_segments (bool, optional): Whether to process in segments. Defaults to True.
#             segment_duration (float, optional): Duration of each segment. Defaults to 3.0.
#             overlap (float, optional): Overlap between segments. Defaults to 0.5.
#             smooth (bool, optional): Whether to apply smoothing. Defaults to True.
            
#         Returns:
#             str: Formatted tablature
#         """
#         print(f"Generating tablature for: {audio_file}")
        
#         if not use_segments:
#             # Process entire audio at once
#             spectrogram_path = self.audio_to_cqt_image(audio_file)
#             tablature = self.predict_tablature(spectrogram_path)
#             formatted_tab = self.format_tablature([tablature])
            
#             # Clean up temporary files
#             if os.path.exists(spectrogram_path):
#                 os.remove(spectrogram_path)
                
#         else:
#             # Process audio in segments
#             segments, sr = self.segment_audio(audio_file, segment_duration, overlap=overlap)
#             tablature_segments = []
#             segment_times = []
            
#             print(f"Processing {len(segments)} segments...")
            
#             # Process each segment
#             for i, (segment, start_time) in enumerate(tqdm(segments)):
#                 # Save segment as temporary audio file
#                 temp_audio = os.path.join("temp_spectrograms", f"temp_segment_{i}.wav")
#                 sf.write(temp_audio, segment, sr)
                
#                 # Convert to spectrogram and predict
#                 spectrogram_path = self.audio_to_cqt_image(temp_audio)
#                 tab = self.predict_tablature(spectrogram_path)
#                 tablature_segments.append(tab)
#                 segment_times.append(start_time)
                
#                 # Clean up temporary files
#                 if os.path.exists(temp_audio):
#                     os.remove(temp_audio)
#                 if os.path.exists(spectrogram_path):
#                     os.remove(spectrogram_path)
            
#             # Apply post-processing if needed
#             if smooth and len(tablature_segments) > 1:
#                 tablature_segments = self.post_process_tablature(tablature_segments)
            
#             # Format the tablature
#             formatted_tab = self.format_tablature(tablature_segments, segment_times)
        
#         # Save to file if specified
#         if output_file:
#             with open(output_file, "w") as f:
#                 f.write(formatted_tab)
#             print(f"\nTablature saved to {output_file}")
        
#         return formatted_tab

# def main():
#     """
#     Main function to parse arguments and generate tablature
#     """
#     parser = argparse.ArgumentParser(description='Generate guitar tablature from audio')
#     parser.add_argument('audio_file', type=str, help='Path to the audio file')
#     parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
#     parser.add_argument('--output', type=str, default=None, help='Output file path')
#     parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to use')
#     parser.add_argument('--no-segments', action='store_true', help='Process audio as a whole (not recommended for long files)')
#     parser.add_argument('--segment-duration', type=float, default=3.0, help='Duration of each segment in seconds')
#     parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between segments (0-1)')
#     parser.add_argument('--no-smooth', action='store_true', help='Disable smoothing')
    
#     args = parser.parse_args()
    
#     # Set output file if not specified
#     if args.output is None:
#         base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
#         args.output = f"{base_name}_tablature.txt"
    
#     # Create generator and generate tablature
#     generator = TablatureGenerator(args.model, args.device)
    
#     tablature = generator.generate_tablature(
#         audio_file=args.audio_file,
#         output_file=args.output,
#         use_segments=not args.no_segments,
#         segment_duration=args.segment_duration,
#         overlap=args.overlap,
#         smooth=not args.no_smooth
#     )
    
#     # Print the tablature
#     print("\nGenerated Tablature:\n")
#     print(tablature)
    
#     # Clean up temp directory
#     for file in os.listdir("temp_spectrograms"):
#         os.remove(os.path.join("temp_spectrograms", file))
#     os.rmdir("temp_spectrograms")

# if __name__ == "__main__":
#     main()
