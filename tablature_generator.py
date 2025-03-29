import torch
import torch.nn as nn
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import warnings
from tqdm import tqdm
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore")

class GuitarTabNet(nn.Module):
    """
    Recreation of the GuitarTabNet model architecture
    """
    def __init__(self, input_channels=3, num_frets=19):
        super(GuitarTabNet, self).__init__()

        # Load ResNet18 and modify first conv layer to accept RGB images
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 256) 
        
        # Separate fully connected layers for each string
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, num_frets)
            ) for _ in range(6)
        ])

    def forward(self, x):
        # Feature extraction with ResNet
        features = self.resnet(x)  
        
        # Apply each string branch
        outputs = [branch(features) for branch in self.branches]
        return outputs

class TablatureGenerator:
    """
    A class to handle the generation of guitar tablature from audio files
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the tablature generator
        
        Args:
            model_path (str): Path to the .pt model file
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to None.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Set up image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create temp directory for spectrograms
        os.makedirs("temp_spectrograms", exist_ok=True)
    
    def _load_model(self, model_path):
        """
        Load the model from file
        
        Args:
            model_path (str): Path to the .pt model file
            
        Returns:
            nn.Module: Loaded model
        """
        try:
            model = GuitarTabNet(input_channels=3, num_frets=19).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check what's in the checkpoint
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Handle both DataParallel and regular models
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the checkpoint is directly a state dict
                state_dict = checkpoint
            
            # Check if it's a DataParallel model
            if list(state_dict.keys())[0].startswith('module.'):
                model = nn.DataParallel(model)
            
            model.load_state_dict(state_dict)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def audio_to_cqt_image(self, audio_file, output_path=None, sr=22050, hop_length=512):
        """
        Convert audio file to CQT spectrogram image
        
        Args:
            audio_file (str): Path to audio file
            output_path (str, optional): Path to save the spectrogram. Defaults to None.
            sr (int, optional): Sample rate. Defaults to 22050.
            hop_length (int, optional): Hop length. Defaults to 512.
            
        Returns:
            str: Path to saved spectrogram image
        """
        if output_path is None:
            output_path = os.path.join("temp_spectrograms", f"{os.path.basename(audio_file)}_spectrogram.png")
        
        # Load audio file
        print(f"Loading audio file: {audio_file}")
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Calculate CQT spectrogram
        C = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C2'), 
                      n_bins=84, bins_per_octave=12)
        
        # Convert to magnitude and apply log transform
        C_mag = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # Create spectrogram image
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(C_mag, sr=sr, x_axis='time', y_axis='cqt_note', 
                               hop_length=hop_length)
        plt.tight_layout()
        
        # Save as image
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved CQT spectrogram to {output_path}")
        return output_path
    
    def segment_audio(self, audio_file, segment_duration=3.0, sr=22050, overlap=0.5):
        """
        Split audio into segments for processing
        
        Args:
            audio_file (str): Path to audio file
            segment_duration (float, optional): Duration of each segment in seconds. Defaults to 3.0.
            sr (int, optional): Sample rate. Defaults to 22050.
            overlap (float, optional): Overlap between segments (0-1). Defaults to 0.5.
            
        Returns:
            tuple: (List of segments, sample rate)
        """
        y, sr = librosa.load(audio_file, sr=sr)
        segment_length = int(segment_duration * sr)
        hop_length = int(segment_length * (1 - overlap))
        segments = []
        
        # Split audio into segments with overlap
        for start in range(0, len(y), hop_length):
            end = min(start + segment_length, len(y))
            segment = y[start:end]
            
            # If segment is too short, pad it
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))
                
            segments.append((segment, start / sr))  # Store segment and its start time
        
        return segments, sr
    
    def predict_tablature(self, image_path):
        """
        Run inference on spectrogram image to predict tablature
        
        Args:
            image_path (str): Path to spectrogram image
            
        Returns:
            list: Predicted fret positions for each string
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        # Convert outputs to tablature format
        tablature = []
        for output in outputs:
            _, predicted_fret = torch.max(output, 1)
            tablature.append(predicted_fret.item())
        
        return tablature
    
    def format_tablature(self, tablature_segments, timings=None):
        """
        Convert model predictions to readable guitar tablature format
        
        Args:
            tablature_segments (list): List of tablature segments
            timings (list, optional): List of segment start times. Defaults to None.
            
        Returns:
            str: Formatted tablature notation
        """
        # Initialize strings for tablature
        strings = [[] for _ in range(6)]  # 6 guitar strings
        
        # Add time markers if available
        time_markers = []
        if timings:
            for t in timings:
                time_markers.append(f"{t:.1f}s")
        
        # Process each segment prediction
        for segment in tablature_segments:
            # Add predictions to each string (in reverse order for standard tablature notation)
            for i, fret in enumerate(reversed(segment)):
                strings[i].append(str(fret))
        
        # Format the tablature
        tab_text = ""
        string_names = ["e|", "B|", "G|", "D|", "A|", "E|"]  # Standard tuning
        
        # Add time markers if available
        if time_markers:
            tab_text += "  " + "  ".join(time_markers) + "\n"
        
        for i, string in enumerate(strings):
            tab_line = string_names[i]
            for fret in string:
                # Format each fret number with consistent spacing
                if len(fret) == 1:
                    tab_line += f"{fret}--"
                else:
                    tab_line += f"{fret}-"
            tab_line += "|"
            tab_text += tab_line + "\n"
        
        return tab_text
    
    def post_process_tablature(self, raw_tablature, smooth_window=3):
        """
        Apply post-processing to the raw tablature predictions
        
        Args:
            raw_tablature (list): List of raw tablature segments
            smooth_window (int, optional): Window size for smoothing. Defaults to 3.
            
        Returns:
            list: Processed tablature
        """
        # If not enough segments for smoothing, return raw
        if len(raw_tablature) <= smooth_window:
            return raw_tablature
            
        processed = []
        
        # Convert to numpy for easier processing
        tab_array = np.array(raw_tablature)
        
        # Simple median filter for each string
        for i in range(tab_array.shape[1]):  # For each string
            string_data = tab_array[:, i]
            
            # Apply median filter
            for j in range(len(string_data)):
                start = max(0, j - smooth_window // 2)
                end = min(len(string_data), j + smooth_window // 2 + 1)
                window = string_data[start:end]
                
                # Only smooth if there's variance in the window
                if np.var(window) > 0:
                    # Use mode (most common value) for smoothing discrete fret numbers
                    values, counts = np.unique(window, return_counts=True)
                    string_data[j] = values[np.argmax(counts)]
            
            # Update the processed array
            tab_array[:, i] = string_data
        
        # Convert back to list format
        for i in range(tab_array.shape[0]):
            processed.append(tab_array[i].tolist())
        
        return processed
    
    def generate_tablature(self, audio_file, output_file=None, use_segments=True, 
                           segment_duration=3.0, overlap=0.5, smooth=True):
        """
        Generate guitar tablature from an audio file
        
        Args:
            audio_file (str): Path to audio file
            output_file (str, optional): Path to save the tablature. Defaults to None.
            use_segments (bool, optional): Whether to process in segments. Defaults to True.
            segment_duration (float, optional): Duration of each segment. Defaults to 3.0.
            overlap (float, optional): Overlap between segments. Defaults to 0.5.
            smooth (bool, optional): Whether to apply smoothing. Defaults to True.
            
        Returns:
            str: Formatted tablature
        """
        print(f"Generating tablature for: {audio_file}")
        
        if not use_segments:
            # Process entire audio at once
            spectrogram_path = self.audio_to_cqt_image(audio_file)
            tablature = self.predict_tablature(spectrogram_path)
            formatted_tab = self.format_tablature([tablature])
            
            # Clean up temporary files
            if os.path.exists(spectrogram_path):
                os.remove(spectrogram_path)
                
        else:
            # Process audio in segments
            segments, sr = self.segment_audio(audio_file, segment_duration, overlap=overlap)
            tablature_segments = []
            segment_times = []
            
            print(f"Processing {len(segments)} segments...")
            
            # Process each segment
            for i, (segment, start_time) in enumerate(tqdm(segments)):
                # Save segment as temporary audio file
                temp_audio = os.path.join("temp_spectrograms", f"temp_segment_{i}.wav")
                sf.write(temp_audio, segment, sr)
                
                # Convert to spectrogram and predict
                spectrogram_path = self.audio_to_cqt_image(temp_audio)
                tab = self.predict_tablature(spectrogram_path)
                tablature_segments.append(tab)
                segment_times.append(start_time)
                
                # Clean up temporary files
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
                if os.path.exists(spectrogram_path):
                    os.remove(spectrogram_path)
            
            # Apply post-processing if needed
            if smooth and len(tablature_segments) > 1:
                tablature_segments = self.post_process_tablature(tablature_segments)
            
            # Format the tablature
            formatted_tab = self.format_tablature(tablature_segments, segment_times)
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(formatted_tab)
            print(f"\nTablature saved to {output_file}")
        
        return formatted_tab

def main():
    """
    Main function to parse arguments and generate tablature
    """
    parser = argparse.ArgumentParser(description='Generate guitar tablature from audio')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--no-segments', action='store_true', help='Process audio as a whole (not recommended for long files)')
    parser.add_argument('--segment-duration', type=float, default=3.0, help='Duration of each segment in seconds')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between segments (0-1)')
    parser.add_argument('--no-smooth', action='store_true', help='Disable smoothing')
    
    args = parser.parse_args()
    
    # Set output file if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        args.output = f"{base_name}_tablature.txt"
    
    # Create generator and generate tablature
    generator = TablatureGenerator(args.model, args.device)
    
    tablature = generator.generate_tablature(
        audio_file=args.audio_file,
        output_file=args.output,
        use_segments=not args.no_segments,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        smooth=not args.no_smooth
    )
    
    # Print the tablature
    print("\nGenerated Tablature:\n")
    print(tablature)
    
    # Clean up temp directory
    for file in os.listdir("temp_spectrograms"):
        os.remove(os.path.join("temp_spectrograms", file))
    os.rmdir("temp_spectrograms")

if __name__ == "__main__":
    main()
