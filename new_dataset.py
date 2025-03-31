import os
import json
import numpy as np
import pandas as pd
import librosa
import jams
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

class GuitarTablatureExtractor:
    def __init__(self, jams_dir, audio_dir, cqt_images_dir, output_dir):
        """
        Initialize the extractor with directory paths
        
        Parameters:
        -----------
        jams_dir : str
            Directory containing JAMS annotation files
        audio_dir : str
            Directory containing audio files (WAV format)
        cqt_images_dir : str
            Directory containing pre-generated CQT image segments
        output_dir : str
            Directory to save the extracted tablature data
        """
        self.jams_dir = Path(jams_dir)
        self.audio_dir = Path(audio_dir)
        self.cqt_images_dir = Path(cqt_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Guitar parameters
        self.num_strings = 6
        self.num_frets = 19  # Including open string (fret 0)
        
        # Standard guitar tuning (EADGBE) in MIDI numbers
        self.open_string_pitches = [40, 45, 50, 55, 59, 64]
        
    def midi_to_tablature(self, midi_pitches, confidence=None):
        """
        Convert MIDI pitch to guitar tablature representation
        
        Parameters:
        -----------
        midi_pitches : list
            List of MIDI pitch values
        confidence : list, optional
            Confidence values for each pitch
            
        Returns:
        --------
        tablature : numpy.ndarray
            Binary tablature representation (6×19)
        """
        # Initialize empty tablature (strings × frets)
        tablature = np.zeros((self.num_strings, self.num_frets), dtype=np.int8)
        
        if len(midi_pitches) == 0:
            return tablature
            
        # Process each MIDI pitch
        for i, pitch in enumerate(midi_pitches):
            conf = confidence[i] if confidence is not None else 1.0
            
            # Skip if confidence is too low
            if conf < 0.5:
                continue
                
            # Find possible string-fret combinations
            possible_positions = []
            for string_idx, open_pitch in enumerate(self.open_string_pitches):
                fret = int(round(pitch - open_pitch))
                # Check if valid fret position
                if 0 <= fret < self.num_frets:
                    possible_positions.append((string_idx, fret))
            
            # Choose the most probable position (prefer lower frets)
            if possible_positions:
                possible_positions.sort(key=lambda x: x[1])
                string_idx, fret = possible_positions[0]
                tablature[string_idx, fret] = 1
                
        return tablature
    
    def extract_tablature_from_jams(self, jams_file, segment_time):
        """
        Extract tablature data for a specific time segment from JAMS file
        
        Parameters:
        -----------
        jams_file : str
            Path to JAMS file
        segment_time : float
            Time point in seconds to extract tablature for
            
        Returns:
        --------
        tablature : numpy.ndarray
            Binary tablature representation (6×19)
        """
        jam = jams.load(jams_file)
        
        # Find relevant note annotations
        midi_notes = []
        midi_conf = []
        
        # Look for note_midi namespace
        for ann in jam.annotations:
            if ann.namespace == 'note_midi':
                for note in ann.data:
                    # Check if the note is active at the segment time
                    start_time = note.time
                    end_time = start_time + note.duration
                    
                    if start_time <= segment_time < end_time:
                        midi_notes.append(note.value)
                        midi_conf.append(1.0)  # Default confidence
                        
        return self.midi_to_tablature(midi_notes, midi_conf)
    
    def extract_tablature_from_pitch_contour(self, jams_file, segment_time):
        """
        Alternative method using pitch contour when note_midi isn't available
        
        Parameters as above
        """
        jam = jams.load(jams_file)
        
        # Find pitch contours near the segment time
        pitches = []
        confidences = []
        
        for ann in jam.annotations:
            if ann.namespace == 'pitch_contour':
                for pitch_obs in ann.data:
                    # Consider pitch observations close to the segment time (within 50ms)
                    if abs(pitch_obs.time - segment_time) < 0.05:
                        # Convert Hz to MIDI
                        if pitch_obs.value > 0:  # Skip silent regions
                            midi_pitch = librosa.hz_to_midi(pitch_obs.value)
                            pitches.append(midi_pitch)
                            confidences.append(pitch_obs.confidence)
        
        return self.midi_to_tablature(pitches, confidences)
    
    def get_cqt_segment_times(self, audio_file, segment_duration=0.2):
        """
        Calculate the center times for each CQT segment
        
        Parameters:
        -----------
        audio_file : str
            Path to audio file
        segment_duration : float
            Duration of each segment in seconds
            
        Returns:
        --------
        segment_times : list
            List of center times for each segment
        """
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate number of segments and their center times
        num_segments = int(duration / segment_duration)
        segment_times = [(i + 0.5) * segment_duration for i in range(num_segments)]
        
        return segment_times
    
    def process_file(self, jams_file, audio_file, segment_duration=0.2):
        """
        Process a complete file, extracting tablature for each segment
        
        Parameters:
        -----------
        jams_file : str
            Path to JAMS file
        audio_file : str
            Path to audio file
        segment_duration : float
            Duration of each segment in seconds
        """
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Get segment times
        segment_times = self.get_cqt_segment_times(audio_file, segment_duration)
        
        # Create output directory for this file
        file_output_dir = self.output_dir / base_name
        file_output_dir.mkdir(exist_ok=True)
        
        tablature_stats = {
            'total': 0,
            'with_notes': 0,
            'with_first_string': 0
        }
        
        # Process each segment
        for i, segment_time in enumerate(segment_times):
            # Check if corresponding CQT image exists
            cqt_image_path = self.cqt_images_dir / f"{base_name}_{i:04d}.png"
            
            if not cqt_image_path.exists():
                print(f"Warning: CQT image not found for {cqt_image_path}")
                continue
            
            # Extract tablature for this segment
            try:
                tablature = self.extract_tablature_from_jams(jams_file, segment_time)
                # If no notes found in tablature, try pitch contour method
                if np.sum(tablature) == 0:
                    tablature = self.extract_tablature_from_pitch_contour(jams_file, segment_time)
            except Exception as e:
                print(f"Error processing {jams_file} at time {segment_time}: {e}")
                continue
            
            # Save tablature as numpy array
            tablature_path = file_output_dir / f"{base_name}_{i:04d}.npy"
            np.save(tablature_path, tablature)
            
            # Update statistics
            tablature_stats['total'] += 1
            if np.sum(tablature) > 0:
                tablature_stats['with_notes'] += 1
            if np.sum(tablature[0, :]) > 0:
                tablature_stats['with_first_string'] += 1
        
        return tablature_stats
    
    def process_all_files(self, segment_duration=0.2):
        """
        Process all files in the directories
        
        Parameters:
        -----------
        segment_duration : float
            Duration of each segment in seconds
        """
        # Get all JAMS files
        jams_files = list(self.jams_dir.glob("*.jams"))
        
        all_stats = {
            'total': 0,
            'with_notes': 0,
            'with_first_string': 0
        }
        
        for jams_file in jams_files:
            base_name = os.path.splitext(jams_file.name)[0]
            
            # Find corresponding audio file (try different formats)
            audio_file = None
            for ext in ['.wav']:
                for prefix in ['hex_debleeded_', 'hex_debleeded-', 'hex_debleeded', '']:
                    potential_file = self.audio_dir / f"{prefix}{base_name}{ext}"
                    if potential_file.exists():
                        audio_file = potential_file
                        break
                if audio_file:
                    break
            
            if not audio_file:
                print(f"Audio file not found for {base_name}")
                continue
            
            print(f"Processing {base_name}")
            stats = self.process_file(jams_file, audio_file, segment_duration)
            
            # Update overall statistics
            for key in all_stats:
                all_stats[key] += stats[key]
                
        print(f"Processing complete. Statistics:")
        print(f"Total tablature files: {all_stats['total']}")
        print(f"Files with any notes: {all_stats['with_notes']}")
        print(f"Files with any notes on first string: {all_stats['with_first_string']}")
        
        return all_stats
    
    def validate_tablature_data(self):
        """
        Validate the tablature data to ensure proper extraction
        """
        # Count files and analyze statistics
        tablature_files = list(self.output_dir.rglob("*.npy"))
        
        if not tablature_files:
            print("No tablature files found!")
            return
        
        print(f"Found {len(tablature_files)} tablature files")
        
        # Load some files to check their content
        sample_count = min(100, len(tablature_files))
        samples = np.random.choice(tablature_files, sample_count, replace=False)
        
        stats = {
            'empty': 0,
            'with_notes': 0,
            'with_first_string': 0,
            'avg_notes_per_tab': 0
        }
        
        note_counts = []
        
        for file_path in samples:
            tab = np.load(file_path)
            note_count = np.sum(tab)
            note_counts.append(note_count)
            
            if note_count == 0:
                stats['empty'] += 1
            else:
                stats['with_notes'] += 1
                
            if np.sum(tab[0, :]) > 0:
                stats['with_first_string'] += 1
        
        stats['avg_notes_per_tab'] = np.mean(note_counts)
        
        print("Validation statistics:")
        print(f"Empty tablatures: {stats['empty']} ({stats['empty']/sample_count*100:.1f}%)")
        print(f"Tablatures with notes: {stats['with_notes']} ({stats['with_notes']/sample_count*100:.1f}%)")
        print(f"Tablatures with first string played: {stats['with_first_string']} ({stats['with_first_string']/sample_count*100:.1f}%)")
        print(f"Average notes per tablature: {stats['avg_notes_per_tab']:.2f}")
        
        # Visualize a few examples
        self.visualize_random_examples(5)
        
        return stats
        
    def visualize_random_examples(self, count=5):
        """
        Visualize random examples of tablature with corresponding CQT images
        
        Parameters:
        -----------
        count : int
            Number of examples to visualize
        """
        tablature_files = list(self.output_dir.rglob("*.npy"))
        
        if len(tablature_files) == 0:
            print("No tablature files to visualize")
            return
            
        samples = np.random.choice(tablature_files, min(count, len(tablature_files)), replace=False)
        
        for tab_path in samples:
            # Get corresponding CQT image
            tab_filename = tab_path.name
            base_name = tab_path.parent.name
            
            cqt_path = self.cqt_images_dir / tab_filename.replace('.npy', '.png')
            if not cqt_path.exists():
                print(f"CQT image not found for {tab_filename}")
                continue
                
            # Load tablature and CQT image
            tab = np.load(tab_path)
            try:
                cqt_img = plt.imread(cqt_path)
            except:
                print(f"Could not read CQT image {cqt_path}")
                continue
                
            # Visualize
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot CQT
            axes[0].imshow(cqt_img)
            axes[0].set_title("CQT Image")
            axes[0].axis('off')
            
            # Plot tablature
            axes[1].imshow(tab, cmap='Blues', interpolation='nearest')
            axes[1].set_title("Tablature (6 strings × 19 frets)")
            axes[1].set_xlabel("Fret number")
            axes[1].set_ylabel("String number")
            
            string_labels = ['E', 'A', 'D', 'G', 'B', 'e']
            axes[1].set_yticks(range(6))
            axes[1].set_yticklabels(string_labels)
            
            plt.tight_layout()
            plt.show()

    def fix_tablature_data(self):
        """
        Attempt to fix issues with tablature data
        """
        print("Analyzing tablature files to find and fix issues...")
        
        tablature_files = list(self.output_dir.rglob("*.npy"))
        if not tablature_files:
            print("No tablature files found!")
            return
        
        tab_with_played_strings = 0
        total_fixed = 0
        
        for file_path in tablature_files:
            tab = np.load(file_path)
            
            # Check if any strings are played
            if np.sum(tab) > 0:
                tab_with_played_strings += 1
                continue
                
            # Try to fix this tablature by inferring from similar segments
            base_name = file_path.parent.name
            segment_id = int(file_path.stem.split('_')[-1])
            
            # Look for neighboring segments (within ±3 segments)
            neighboring_tabs = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                neighbor_id = segment_id + offset
                if neighbor_id < 0:
                    continue
                    
                neighbor_path = file_path.parent / f"{base_name}_{neighbor_id:04d}.npy"
                if neighbor_path.exists():
                    neighbor_tab = np.load(neighbor_path)
                    if np.sum(neighbor_tab) > 0:
                        neighboring_tabs.append(neighbor_tab)
            
            # If we found neighboring segments with notes, use them to infer
            if neighboring_tabs:
                # Simple inference: use the most common note pattern from neighbors
                combined_tab = np.zeros_like(tab)
                for neighbor_tab in neighboring_tabs:
                    combined_tab += neighbor_tab
                
                # Apply threshold to get most common notes
                threshold = len(neighboring_tabs) / 2
                inferred_tab = np.zeros_like(tab)
                inferred_tab[combined_tab > threshold] = 1
                
                # Only use the inference if it has at least one note
                if np.sum(inferred_tab) > 0:
                    np.save(file_path, inferred_tab)
                    total_fixed += 1
        
        print(f"Analysis complete:")
        print(f"Total tablature files: {len(tablature_files)}")
        print(f"Files with played strings: {tab_with_played_strings} ({tab_with_played_strings/len(tablature_files)*100:.1f}%)")
        print(f"Empty files fixed through inference: {total_fixed}")
        
        return {
            'total': len(tablature_files),
            'with_played_strings': tab_with_played_strings,
            'fixed': total_fixed
        }


# Example usage
if __name__ == "__main__":
    # Set up paths
    JAMS_DIR = "path/to/jams_files"
    AUDIO_DIR = "path/to/audio_files"
    CQT_IMAGES_DIR = "path/to/cqt_images"
    OUTPUT_DIR = "path/to/output_tablature"
    
    # Initialize and run the extractor
    extractor = GuitarTablatureExtractor(JAMS_DIR, AUDIO_DIR, CQT_IMAGES_DIR, OUTPUT_DIR)
    
    # Process all files
    extractor.process_all_files(segment_duration=0.2)
    
    # Validate the extracted data
    stats = extractor.validate_tablature_data()
    
    # If there's an issue with too many empty tablatures, try to fix it
    if stats['empty'] / (stats['empty'] + stats['with_notes']) > 0.8:
        print("Too many empty tablatures detected. Attempting to fix...")
        extractor.fix_tablature_data()
        
        # Validate again after fixing
        updated_stats = extractor.validate_tablature_data()
        
        # If still problematic, try more aggressive methods
        if updated_stats['empty'] / (updated_stats['empty'] + updated_stats['with_notes']) > 0.7:
            print("Still too many empty tablatures. Consider revising the extraction method.")
