# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import librosa
# import argparse
# from ViT_model import ViTGuitarTabModel
# from transformers import ViTImageProcessor

# # Constants
# SAMPLE_RATE = 22050
# HOP_LENGTH = 512
# N_BINS = 84
# BINS_PER_OCTAVE = 12
# FMIN = librosa.note_to_hz('C1')
# FRAME_LENGTH = 2048  # samples
# TUNING = ['E', 'A', 'D', 'G', 'B', 'E']  # Standard guitar tuning (low to high)
# FRET_LABELS = [str(i) for i in range(19)]  # 0-18 frets

# class TabGenerator:
#     def __init__(self, model_path, device=None):
#         # Set device
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device
        
#         print(f"Using device: {self.device}")
        
#         # Load model
#         self.model = ViTGuitarTabModel(num_classes=19, pretrained_model="facebook/dino-vits8")
#         checkpoint = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.to(self.device)
#         self.model.eval()
        
#         # Initialize ViT image processor
#         self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8")
        
#         print("Model loaded successfully")

#     def preprocess_audio(self, audio_path, segment_duration=1.0):
#         """
#         Load and preprocess audio file to CQT segments
#         Returns segments as normalized tensors ready for model input
#         """
#         print(f"Loading audio: {audio_path}")
        
#         # Load audio file
#         y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
#         # Calculate segment length in samples
#         segment_samples = int(segment_duration * sr)
        
#         # Split audio into segments
#         segments = []
#         for start in range(0, len(y), segment_samples):
#             end = min(start + segment_samples, len(y))
#             segment = y[start:end]
            
#             # Zero-pad if segment is shorter than expected
#             if len(segment) < segment_samples:
#                 segment = np.pad(segment, (0, segment_samples - len(segment)))
            
#             # Compute CQT
#             C = librosa.cqt(
#                 segment, 
#                 sr=sr, 
#                 hop_length=HOP_LENGTH,
#                 n_bins=N_BINS,
#                 bins_per_octave=BINS_PER_OCTAVE,
#                 fmin=FMIN
#             )
            
#             # Convert to dB scale
#             C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
            
#             # Normalize to [0, 1] range
#             C_normalized = (C_db + 120) / 120  # Assuming minimum is -120 dB
#             C_normalized = np.clip(C_normalized, 0, 1)
            
#             segments.append(C_normalized)
        
#         # Convert to tensors and prepare for ViT
#         tensor_segments = []
#         for segment in segments:
#             # Convert to tensor
#             tensor = torch.tensor(segment, dtype=torch.float32)
            
#             # Resize to 224x224 for ViT
#             tensor = tensor.unsqueeze(0)  # Add channel dimension: [1, H, W]
#             tensor = torch.nn.functional.interpolate(
#                 tensor.unsqueeze(0),  # Add batch dimension
#                 size=(224, 224),
#                 mode='bicubic',
#                 align_corners=False
#             ).squeeze(0)  # Remove batch dimension
            
#             # Repeat to 3 channels
#             tensor = tensor.repeat(3, 1, 1)  # [3, 224, 224]
            
#             tensor_segments.append(tensor)
        
#         print(f"Created {len(tensor_segments)} segments from audio")
#         return tensor_segments

#     def predict_tablature(self, tensor_segments):
#         """
#         Generate tablature predictions for each audio segment
#         Returns list of predictions for each segment
#         """
#         all_predictions = []
        
#         with torch.no_grad():
#             for segment in tensor_segments:
#                 # Prepare input
#                 inputs = segment.unsqueeze(0).to(self.device)  # Add batch dimension: [1, 3, 224, 224]
                
#                 # Process with ViT
#                 processed_inputs = self.processor(images=inputs, return_tensors="pt").pixel_values.to(self.device)
                
#                 # Get model predictions
#                 outputs = self.model(processed_inputs)
                
#                 # Extract predictions for each string
#                 segment_predictions = []
#                 for string_output in outputs:
#                     _, predicted = torch.max(string_output.data, 1)
#                     segment_predictions.append(predicted.item())
                
#                 all_predictions.append(segment_predictions)
        
#         return all_predictions

#     def create_tablature_image(self, predictions, output_path, segment_duration=1.0):
#         """
#         Create a tablature image from predictions and save as PNG
#         """
#         num_segments = len(predictions)
        
#         # Create figure and axis
#         fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
#         # Hide axes
#         ax.axis('off')
        
#         # Title
#         plt.title('Generated Guitar Tablature', fontsize=16, fontweight='bold')
        
#         # Set up tablature text
#         tab_lines = []
#         for string_idx in range(6):
#             string_name = TUNING[string_idx]
#             line = f"{string_name}|"
            
#             for segment_idx in range(num_segments):
#                 fret = predictions[segment_idx][string_idx]
                
#                 # Format fret number (pad with spaces)
#                 if fret < 10:
#                     fret_str = f"{fret}-"
#                 else:
#                     fret_str = f"{fret}"
                
#                 # Add timing separation every 4 segments
#                 if (segment_idx + 1) % 4 == 0 and segment_idx < num_segments - 1:
#                     line += fret_str + "|"
#                 else:
#                     line += fret_str + "-"
            
#             line += "|"
#             tab_lines.append(line)
        
#         # Add time markers
#         time_line = "  "
#         for i in range(num_segments):
#             if i % 4 == 0:
#                 seconds = i * segment_duration
#                 time_marker = f"{seconds:.1f}s"
#                 padding = "-" * (4 - len(time_marker))
#                 time_line += time_marker + padding
        
#         # Draw tablature
#         tab_text = "\n".join(tab_lines)
#         plt.text(0.1, 0.5, tab_text, fontfamily='monospace', fontsize=12, verticalalignment='center')
        
#         # Add time markers at bottom
#         plt.text(0.1, 0.2, time_line, fontfamily='monospace', fontsize=10)
        
#         # Add legend/key
#         legend_text = "Legend: Numbers represent frets (0 = open string)"
#         plt.text(0.1, 0.1, legend_text, fontsize=10)
        
#         # Save image
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"Tablature saved to {output_path}")
#         return output_path

#     def generate_tab(self, audio_path, output_path=None):
#         """
#         Main function to generate tablature from audio file
#         """
#         # Set default output path if not provided
#         if output_path is None:
#             base_name = os.path.splitext(os.path.basename(audio_path))[0]
#             output_path = f"{base_name}_tablature.png"
        
#         # Process audio
#         tensor_segments = self.preprocess_audio(audio_path)
        
#         # Generate predictions
#         predictions = self.predict_tablature(tensor_segments)
        
#         # Create tablature image
#         tab_path = self.create_tablature_image(predictions, output_path)
        
#         return tab_path

# def main():
#     parser = argparse.ArgumentParser(description='Guitar Tablature Generator')
#     parser.add_argument('--input', '-i', required=True, help='Path to input MP3 file')
#     parser.add_argument('--output', '-o', help='Path to output PNG file (optional)')
#     parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint file')
    
#     args = parser.parse_args()
    
#     # Create tablature generator
#     tab_generator = TabGenerator(model_path=args.model)
    
#     # Generate tablature
#     tab_generator.generate_tab(args.input, args.output)

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import argparse
from transformers import ViTImageProcessor
from ViT_model import ViTGuitarTabModel

class TablatureGenerator:
    def __init__(self, model_path, device=None):
        """
        Initialize the tablature generator with a trained model.
        
        Args:
            model_path: Path to the trained model checkpoint (.pt file)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = ViTGuitarTabModel(num_classes=19, pretrained_model="facebook/dino-vits8")
        self.model.to(self.device)
        
        # Load the model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best validation accuracy: {checkpoint.get('accuracies', 'N/A')}")
        
        # Map fret numbers to actual fret values (0-18)
        self.fret_mapping = {i: i for i in range(19)}  # 0 = open string, 1-18 = frets
        
        # Initialize processor for ViT
        self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8")
        checkpoint = torch.load(model_path, map_location=self.device)
    
    # Rename the state dict keys to match your model
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.'):
            # Replace 'encoder.' with 'vit.'
                new_key = key.replace('encoder.', 'vit.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
    
    # Try to load with strict=False to ignore missing/unexpected keys
        self.model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded with some parameters ignored or missing. This might affect performance.")

    def preprocess_audio(self, audio_file, segment_duration=0.2, hop_duration=0.1):
        """
        Preprocess the audio file into CQT segments.
        
        Args:
            audio_file: Path to the WAV file
            segment_duration: Duration of each segment in seconds
            hop_duration: Hop length between segments in seconds
            
        Returns:
            List of preprocessed CQT segments
        """
        print(f"Loading audio file: {audio_file}")
        
        # Load audio
        data, sr = librosa.load(audio_file, sr=44100, mono=True)
        
        # Calculate segment and hop lengths in samples
        segment_length = int(segment_duration * sr)
        hop_length = int(hop_duration * sr)
        
        # Calculate number of segments
        num_segments = max(1, int((len(data) - segment_length) / hop_length) + 1)
        
        segments = []
        timestamps = []
        
        print(f"Generating CQT for {num_segments} segments...")
        
        for i in range(num_segments):
            # Extract segment
            start_idx = i * hop_length
            end_idx = min(start_idx + segment_length, len(data))
            
            if end_idx - start_idx < segment_length // 2:  # Skip if segment is too short
                continue
            
            segment = data[start_idx:end_idx]
            
            # Pad if necessary
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
            
            # Compute CQT
            CQT = librosa.cqt(segment, sr=sr, hop_length=1024, fmin=None, n_bins=96, bins_per_octave=12)
            CQT_mag = librosa.magphase(CQT)[0]**4
            CQTdB = librosa.core.amplitude_to_db(CQT_mag, ref=np.amax)
            
            # Apply noise limiting (as in your original code)
            new_CQT = self.cqt_lim(CQTdB)
            
            # Normalize CQT to [0, 1] range
            normalized_cqt = (new_CQT + 120) / 120  # Assuming minimum value is -120 dB
            normalized_cqt = np.clip(normalized_cqt, 0, 1)
            
            # Store segment and timestamp
            segments.append(normalized_cqt)
            timestamps.append(start_idx / sr)  # Convert to seconds
        
        return segments, timestamps
    
    def cqt_lim(self, CQT, min_db=-60, silence_db=-120):
        """Limit noise in CQT by setting values below threshold to silence level."""
        new_CQT = np.copy(CQT)
        new_CQT[new_CQT < min_db] = silence_db
        return new_CQT
    
    def prepare_for_vit(self, cqt):
        """Prepare the CQT data for input to the ViT model."""
        # Convert to tensor
        cqt_tensor = torch.tensor(cqt, dtype=torch.float32)
        
        # Add channel dimension if needed
        if len(cqt_tensor.shape) == 2:  # (H, W)
            cqt_tensor = cqt_tensor.unsqueeze(0)  # (1, H, W)
        
        # Resize to 224x224 for ViT
        cqt_tensor = F.interpolate(
            cqt_tensor.unsqueeze(0),  # Add batch dimension
            size=(224, 224),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Convert to 3 channels for ViT
        if cqt_tensor.shape[0] == 1:
            cqt_tensor = cqt_tensor.repeat(3, 1, 1)  # (1, H, W) → (3, H, W)
        
        # Process with ViT processor
        inputs = self.processor(images=cqt_tensor, return_tensors="pt")
        return inputs.pixel_values
    
    def predict_tablature(self, segments):
        """
        Predict tablature from CQT segments.
        
        Args:
            segments: List of preprocessed CQT segments
            
        Returns:
            List of tablature frames, each with 6 fret predictions
        """
        tablature = []
        
        for segment in segments:
            # Prepare for model input
            inputs = self.prepare_for_vit(segment)
            inputs = inputs.to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Process outputs (one for each string)
            segment_tabs = []
            for string_output in outputs:
                # Get the most likely fret
                _, predicted_fret = torch.max(string_output.data, 1)
                segment_tabs.append(self.fret_mapping[predicted_fret.item()])
            
            tablature.append(segment_tabs)
        
        return tablature
    
    def generate_tablature(self, audio_file, output_file=None):
        """
        Generate tablature from an audio file.
        
        Args:
            audio_file: Path to the audio file
            output_file: Path to save the tablature (if None, only returns the tablature)
            
        Returns:
            Tablature as a list of [time, string1, string2, ...] entries
        """
        # Preprocess audio
        segments, timestamps = self.preprocess_audio(audio_file)
        
        if not segments:
            print("No valid segments found in the audio file.")
            return []
        
        # Predict tablature
        print("Predicting tablature...")
        tablature = self.predict_tablature(segments)
        
        # Combine with timestamps
        final_tablature = []
        for i, (time, tab) in enumerate(zip(timestamps, tablature)):
            final_tablature.append([time] + tab)
        
        # Save tablature if output file is specified
        if output_file:
            self.save_tablature(final_tablature, output_file, audio_file)
        
        return final_tablature
    
    def save_tablature(self, tablature, output_file, audio_file):
        """
        Save the tablature to a file.
        
        Args:
            tablature: List of [time, string1, string2, ...] entries
            output_file: Path to save the tablature
            audio_file: Path to the original audio file (for reference)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write tablature to file
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"# Tablature for {os.path.basename(audio_file)}\n")
            f.write(f"# Generated on {torch.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write standard guitar tablature format
            f.write("E|")
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                if e_string == 0:
                    f.write("---|")
                else:
                    f.write(f"{e_string:2d}-|")
            f.write("\n")
            
            f.write("B|")
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                if b_string == 0:
                    f.write("---|")
                else:
                    f.write(f"{b_string:2d}-|")
            f.write("\n")
            
            f.write("G|")
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                if g_string == 0:
                    f.write("---|")
                else:
                    f.write(f"{g_string:2d}-|")
            f.write("\n")
            
            f.write("D|")
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                if d_string == 0:
                    f.write("---|")
                else:
                    f.write(f"{d_string:2d}-|")
            f.write("\n")
            
            f.write("A|")
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                if a_string == 0:
                    f.write("---|")
                else:
                    f.write(f"{a_string:2d}-|")
            f.write("\n")
            
            f.write("E|")
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                if e_string_high == 0:
                    f.write("---|")
                else:
                    f.write(f"{e_string_high:2d}-|")
            f.write("\n")
            
            # Write detailed information (time-based)
            f.write("\n# Detailed Time-Based Tablature:\n")
            f.write("# Time(s) | Low E | A | D | G | B | High E\n")
            
            for entry in tablature:
                time, e_string, a_string, d_string, g_string, b_string, e_string_high = entry
                f.write(f"{time:.2f} | {e_string_high} | {b_string} | {g_string} | {d_string} | {a_string} | {e_string}\n")
        
        print(f"Tablature saved to {output_file}")
    
    def visualize_tablature(self, tablature, output_file=None):
        """
        Visualize the tablature as a plot.
        
        Args:
            tablature: List of [time, string1, string2, ...] entries
            output_file: Path to save the visualization
        """
        times = [entry[0] for entry in tablature]
        strings = np.array([entry[1:] for entry in tablature])
        
        plt.figure(figsize=(12, 8))
        string_names = ['Low E', 'A', 'D', 'G', 'B', 'High E']
        
        # Plot each string
        for i in range(6):
            plt.subplot(6, 1, i+1)
            plt.plot(times, strings[:, i], 'o-', markersize=4)
            plt.ylabel(string_names[i])
            plt.yticks(range(0, 19))
            plt.grid(True, alpha=0.3)
            
            if i == 0:
                plt.title('Guitar Tablature')
            if i == 5:
                plt.xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate guitar tablature from audio file')
    parser.add_argument('audio_file', type=str, help='Path to the audio file (WAV format)')
    parser.add_argument('--model', type=str, default='best_vit_guitar_tab_model.pt', help='Path to the trained model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Path to save the tablature output')
    parser.add_argument('--visualize', action='store_true', help='Visualize the tablature')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file {args.audio_file} does not exist.")
        return
    
    # Check if model file exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file {args.model} does not exist.")
        return
    
    # Generate default output filename if not specified
    if args.output is None:
        base_name = os.path.splitext(args.audio_file)[0]
        args.output = f"{base_name}_tab.txt"
    
    # Initialize tablature generator
    generator = TablatureGenerator(args.model, args.device)
    
    # Generate tablature
    tablature = generator.generate_tablature(args.audio_file, args.output)
    
    # Visualize if requested
    if args.visualize and tablature:
        viz_output = os.path.splitext(args.output)[0] + "_viz.png"
        generator.visualize_tablature(tablature, viz_output)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     main()
