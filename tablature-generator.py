import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import argparse
from ViT_model import ViTGuitarTabModel
from transformers import ViTImageProcessor

# Constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 84
BINS_PER_OCTAVE = 12
FMIN = librosa.note_to_hz('C1')
FRAME_LENGTH = 2048  # samples
TUNING = ['E', 'A', 'D', 'G', 'B', 'E']  # Standard guitar tuning (low to high)
FRET_LABELS = [str(i) for i in range(19)]  # 0-18 frets

class TabGenerator:
    def __init__(self, model_path, device=None):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = ViTGuitarTabModel(num_classes=19, pretrained_model="facebook/dino-vits8")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize ViT image processor
        self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8")
        
        print("Model loaded successfully")

    def preprocess_audio(self, audio_path, segment_duration=1.0):
        """
        Load and preprocess audio file to CQT segments
        Returns segments as normalized tensors ready for model input
        """
        print(f"Loading audio: {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Calculate segment length in samples
        segment_samples = int(segment_duration * sr)
        
        # Split audio into segments
        segments = []
        for start in range(0, len(y), segment_samples):
            end = min(start + segment_samples, len(y))
            segment = y[start:end]
            
            # Zero-pad if segment is shorter than expected
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            
            # Compute CQT
            C = librosa.cqt(
                segment, 
                sr=sr, 
                hop_length=HOP_LENGTH,
                n_bins=N_BINS,
                bins_per_octave=BINS_PER_OCTAVE,
                fmin=FMIN
            )
            
            # Convert to dB scale
            C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
            
            # Normalize to [0, 1] range
            C_normalized = (C_db + 120) / 120  # Assuming minimum is -120 dB
            C_normalized = np.clip(C_normalized, 0, 1)
            
            segments.append(C_normalized)
        
        # Convert to tensors and prepare for ViT
        tensor_segments = []
        for segment in segments:
            # Convert to tensor
            tensor = torch.tensor(segment, dtype=torch.float32)
            
            # Resize to 224x224 for ViT
            tensor = tensor.unsqueeze(0)  # Add channel dimension: [1, H, W]
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),  # Add batch dimension
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
            
            # Repeat to 3 channels
            tensor = tensor.repeat(3, 1, 1)  # [3, 224, 224]
            
            tensor_segments.append(tensor)
        
        print(f"Created {len(tensor_segments)} segments from audio")
        return tensor_segments

    def predict_tablature(self, tensor_segments):
        """
        Generate tablature predictions for each audio segment
        Returns list of predictions for each segment
        """
        all_predictions = []
        
        with torch.no_grad():
            for segment in tensor_segments:
                # Prepare input
                inputs = segment.unsqueeze(0).to(self.device)  # Add batch dimension: [1, 3, 224, 224]
                
                # Process with ViT
                processed_inputs = self.processor(images=inputs, return_tensors="pt").pixel_values.to(self.device)
                
                # Get model predictions
                outputs = self.model(processed_inputs)
                
                # Extract predictions for each string
                segment_predictions = []
                for string_output in outputs:
                    _, predicted = torch.max(string_output.data, 1)
                    segment_predictions.append(predicted.item())
                
                all_predictions.append(segment_predictions)
        
        return all_predictions

    def create_tablature_image(self, predictions, output_path, segment_duration=1.0):
        """
        Create a tablature image from predictions and save as PNG
        """
        num_segments = len(predictions)
        
        # Create figure and axis
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        # Hide axes
        ax.axis('off')
        
        # Title
        plt.title('Generated Guitar Tablature', fontsize=16, fontweight='bold')
        
        # Set up tablature text
        tab_lines = []
        for string_idx in range(6):
            string_name = TUNING[string_idx]
            line = f"{string_name}|"
            
            for segment_idx in range(num_segments):
                fret = predictions[segment_idx][string_idx]
                
                # Format fret number (pad with spaces)
                if fret < 10:
                    fret_str = f"{fret}-"
                else:
                    fret_str = f"{fret}"
                
                # Add timing separation every 4 segments
                if (segment_idx + 1) % 4 == 0 and segment_idx < num_segments - 1:
                    line += fret_str + "|"
                else:
                    line += fret_str + "-"
            
            line += "|"
            tab_lines.append(line)
        
        # Add time markers
        time_line = "  "
        for i in range(num_segments):
            if i % 4 == 0:
                seconds = i * segment_duration
                time_marker = f"{seconds:.1f}s"
                padding = "-" * (4 - len(time_marker))
                time_line += time_marker + padding
        
        # Draw tablature
        tab_text = "\n".join(tab_lines)
        plt.text(0.1, 0.5, tab_text, fontfamily='monospace', fontsize=12, verticalalignment='center')
        
        # Add time markers at bottom
        plt.text(0.1, 0.2, time_line, fontfamily='monospace', fontsize=10)
        
        # Add legend/key
        legend_text = "Legend: Numbers represent frets (0 = open string)"
        plt.text(0.1, 0.1, legend_text, fontsize=10)
        
        # Save image
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tablature saved to {output_path}")
        return output_path

    def generate_tab(self, audio_path, output_path=None):
        """
        Main function to generate tablature from audio file
        """
        # Set default output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"{base_name}_tablature.png"
        
        # Process audio
        tensor_segments = self.preprocess_audio(audio_path)
        
        # Generate predictions
        predictions = self.predict_tablature(tensor_segments)
        
        # Create tablature image
        tab_path = self.create_tablature_image(predictions, output_path)
        
        return tab_path

def main():
    parser = argparse.ArgumentParser(description='Guitar Tablature Generator')
    parser.add_argument('--input', '-i', required=True, help='Path to input MP3 file')
    parser.add_argument('--output', '-o', help='Path to output PNG file (optional)')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint file')
    
    args = parser.parse_args()
    
    # Create tablature generator
    tab_generator = TabGenerator(model_path=args.model)
    
    # Generate tablature
    tab_generator.generate_tab(args.input, args.output)

if __name__ == "__main__":
    main()
