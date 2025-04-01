import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import jams
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GuitarCQTAndTablatureGenerator:
    def __init__(self, audio_dir, jams_dir, cqt_output_dir, tablature_output_dir, segment_duration=0.2):
        """
        Initialize the generator with directory paths
        
        Parameters:
        -----------
        audio_dir : str
            Directory containing audio files (WAV format)
        jams_dir : str
            Directory containing JAMS annotation files
        cqt_output_dir : str
            Directory to save generated CQT images
        tablature_output_dir : str
            Directory to save the extracted tablature data
        segment_duration : float
            Duration of each segment in seconds
        """
        self.audio_dir = Path(audio_dir)
        self.jams_dir = Path(jams_dir)
        self.cqt_output_dir = Path(cqt_output_dir)
        self.tablature_output_dir = Path(tablature_output_dir)
        self.segment_duration = segment_duration
        
        # Create output directories
        self.cqt_output_dir.mkdir(exist_ok=True, parents=True)
        self.tablature_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Guitar parameters
        self.num_strings = 6
        self.num_frets = 19  # Including open string (fret 0)
        
        # Standard guitar tuning (EADGBE) in MIDI numbers
        self.open_string_pitches = [40, 45, 50, 55, 59, 64]
        
        logger.info(f"Audio directory: {self.audio_dir} (exists: {self.audio_dir.exists()})")
        logger.info(f"JAMS directory: {self.jams_dir} (exists: {self.jams_dir.exists()})")
        logger.info(f"CQT output directory: {self.cqt_output_dir}")
        logger.info(f"Tablature output directory: {self.tablature_output_dir}")
        
    def generate_cqt_images(self, audio_path, file_output_dir):
        """
        Generate CQT images for all segments in an audio file
        
        Parameters:
        -----------
        audio_path : Path
            Path to the audio file
        file_output_dir : Path
            Directory to save the CQT images for this file
            
        Returns:
        --------
        segment_times : list
            List of segment center times
        """
        try:
            # Create directory for this file's CQT images
            file_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Get the base filename without extension
            base_name = audio_path.stem
            
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate number of samples per segment
            samples_per_segment = int(self.segment_duration * sr)
            
            # Calculate total number of segments
            duration = librosa.get_duration(y=y, sr=sr)
            num_segments = int(duration / self.segment_duration)
            
            # Generate segment center times (needed for tablature extraction)
            segment_times = [(i + 0.5) * self.segment_duration for i in range(num_segments)]
            
            # Process each segment
            for i in range(num_segments):
                # Get the segment
                start_sample = i * samples_per_segment
                end_sample = min(start_sample + samples_per_segment, len(y))
                segment = y[start_sample:end_sample]
                
                # If the segment is too short, pad it
                if len(segment) < samples_per_segment:
                    segment = np.pad(segment, (0, samples_per_segment - len(segment)), 'constant')
                
                # Calculate CQT
                C = librosa.cqt(segment, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=84, 
                                bins_per_octave=12, hop_length=512)
                
                # Convert to dB
                C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
                
                # Create the figure and plot
                fig, ax = plt.subplots(figsize=(10, 5))
                canvas = FigureCanvas(fig)
                
                img = librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_note', 
                                             fmin=librosa.note_to_hz('C1'), ax=ax)
                
                ax.set_title(f'{base_name} - Segment {i:04d}')
                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                
                # Save the figure
                output_filename = f"{base_name}_{i:04d}.png"
                output_path = file_output_dir / output_filename
                fig.tight_layout()
                fig.savefig(output_path, dpi=100)
                plt.close(fig)
                
            logger.info(f"Generated {num_segments} CQT images for {base_name}")
            return segment_times
        
        except Exception as e:
            logger.error(f"Error generating CQT images for {audio_path}: {str(e)}")
            return []

    def midi_to_tablature(self, midi_pitches, confidence=None):
        """
        Convert MIDI pitch to guitar tablature representation
        
        Parameters:
        -----------
        midi_pitches : list
            List of MIDI pitch values
        confidence : list, optional
            List of confidence values for each pitch
            
        Returns:
        --------
        tablature : numpy.ndarray
            2D array representing the tablature (strings × frets)
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
            
            # Handle dictionary pitch values
            if isinstance(pitch, dict):
                # Try to extract the pitch value from the dictionary
                pitch = pitch.get('pitch') or pitch.get('value')
            
            # Ensure pitch is not None
            if pitch is None:
                continue
                
            # Convert to float safely
            try:
                pitch_value = float(pitch)
            except (ValueError, TypeError):
                continue
                
            # Find possible string-fret combinations
            possible_positions = []
            for string_idx, open_pitch in enumerate(self.open_string_pitches):
                fret = int(round(pitch_value - open_pitch))
                # Check if valid fret position
                if 0 <= fret < self.num_frets:
                    possible_positions.append((string_idx, fret))
            
            # Choose the most probable position (prefer lower frets)
            if possible_positions:
                possible_positions.sort(key=lambda x: x[1])
                string_idx, fret = possible_positions[0]
                tablature[string_idx, fret] = 1
                
        return tablature

    def extract_tablature_from_jams(self, jam, segment_time):
        """
        Extract tablature data from JAMS file at given segment time
        
        Parameters:
        -----------
        jam : jams.JAMS
            JAMS annotation object
        segment_time : float
            Time of the segment in seconds
            
        Returns:
        --------
        tablature : numpy.ndarray
            2D array representing the tablature (strings × frets)
        """
        # Find relevant note annotations
        midi_notes = []
        midi_conf = []
    
        # Look for note_midi namespace
        for ann in jam.annotations:
            if ann.namespace == 'note_midi':
                for note in ann.data:
                    # Check if the note is active at the segment time
                    start_time = note.time
                    end_time = start_time + note.duration if note.duration is not None else None
                    
                    if start_time <= segment_time < end_time:
                        # Handle both dictionary and direct value cases
                        if isinstance(note.value, dict):
                            if 'pitch' in note.value:
                                midi_notes.append(note.value['pitch'])
                            elif 'value' in note.value:
                                midi_notes.append(note.value['value'])
                            else:
                                continue
                        else:
                            midi_notes.append(note.value)
                            
                        midi_conf.append(note.confidence if note.confidence is not None else 1.0)
                    
        return self.midi_to_tablature(midi_notes, midi_conf)

    def extract_tablature_from_pitch_contour(self, jam, segment_time):
        """
        Alternative method using pitch contour when note_midi isn't available
        
        Parameters:
        -----------
        jam : jams.JAMS
            JAMS annotation object
        segment_time : float
            Time of the segment in seconds
            
        Returns:
        --------
        tablature : numpy.ndarray
            2D array representing the tablature (strings × frets)
        """
        # Find pitch contours near the segment time
        pitches = []
        confidences = []
    
        for ann in jam.annotations:
            if ann.namespace == 'pitch_contour':
                for pitch_obs in ann.data:
                    # Consider pitch observations close to the segment time (within 50ms)
                    if abs(pitch_obs.time - segment_time) < 0.05:
                        # Handle both dictionary and direct value cases
                        pitch_val = None
                        if isinstance(pitch_obs.value, dict):
                            if 'frequency' in pitch_obs.value:
                                pitch_val = pitch_obs.value['frequency']
                            elif 'value' in pitch_obs.value:
                                pitch_val = pitch_obs.value['value']
                        else:
                            pitch_val = pitch_obs.value
                            
                        # Convert Hz to MIDI if we have a valid value
                        if pitch_val is not None and pitch_val > 0:  # Skip silent regions
                            try:
                                midi_pitch = librosa.hz_to_midi(float(pitch_val))
                                pitches.append(midi_pitch)
                                confidences.append(pitch_obs.confidence if pitch_obs.confidence is not None else 1.0)
                            except (ValueError, TypeError):
                                # Skip if conversion fails
                                continue
    
        return self.midi_to_tablature(pitches, confidences)
        
    def generate_tablature(self, jams_path, segment_times, file_output_dir, base_name):
        """
        Generate tablature data for all segments in a file
        
        Parameters:
        -----------
        jams_path : Path
            Path to the JAMS file
        segment_times : list
            List of segment center times
        file_output_dir : Path
            Directory to save the tablature data for this file
        base_name : str
            Base name of the file (for output naming)
            
        Returns:
        --------
        success : bool
            True if tablature generation was successful
        """
        try:
            # Create directory for this file's tablature data
            file_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Load JAMS file
            jam = jams.load(str(jams_path))
            
            # Process each segment
            for i, segment_time in enumerate(segment_times):
                # Extract tablature for this segment
                tablature = self.extract_tablature_from_jams(jam, segment_time)
                
                # If no notes found in tablature, try pitch contour method
                if np.sum(tablature) == 0:
                    tablature = self.extract_tablature_from_pitch_contour(jam, segment_time)
                
                # Save tablature as numpy array
                output_filename = f"{base_name}_{i:04d}.npy"
                output_path = file_output_dir / output_filename
                np.save(output_path, tablature)
                
            logger.info(f"Generated {len(segment_times)} tablature arrays for {base_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating tablature data for {jams_path}: {str(e)}")
            return False
            
    def process_file(self, audio_path, jams_path):
        """
        Process a single file, generating both CQT images and tablature data
        
        Parameters:
        -----------
        audio_path : Path
            Path to the audio file
        jams_path : Path
            Path to the JAMS file
            
        Returns:
        --------
        success : bool
            True if both CQT and tablature generation were successful
        """
        try:
            # Get base name for output directories and files
            base_name = audio_path.stem
            
            # Create output directories for this file
            cqt_file_dir = self.cqt_output_dir / base_name
            tab_file_dir = self.tablature_output_dir / base_name
            
            logger.info(f"Processing file: {base_name}")
            
            # Step 1: Generate CQT images
            segment_times = self.generate_cqt_images(audio_path, cqt_file_dir)
            
            if not segment_times:
                logger.error(f"Failed to generate CQT images for {base_name}")
                return False
                
            # Step 2: Generate tablature data
            tab_success = self.generate_tablature(jams_path, segment_times, tab_file_dir, base_name)
            
            if not tab_success:
                logger.error(f"Failed to generate tablature data for {base_name}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {audio_path}: {str(e)}")
            return False
            
    def process_all_files(self, max_workers=4):
        """
        Process all matching audio and JAMS files
        
        Parameters:
        -----------
        max_workers : int
            Maximum number of parallel workers
            
        Returns:
        --------
        stats : dict
            Statistics about the processing
        """
        # Get all audio files
        audio_files = list(self.audio_dir.glob("*.wav"))
        
        if not audio_files:
            logger.error(f"No audio files found in {self.audio_dir}")
            return {'total': 0, 'success': 0, 'error': 0}
            
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Find matching JAMS files
        valid_pairs = []
        
        for audio_path in audio_files:
            base_name = audio_path.stem
            jams_path = self.jams_dir / f"{base_name}.jams"
            
            if jams_path.exists():
                valid_pairs.append((audio_path, jams_path))
            else:
                logger.warning(f"No matching JAMS file found for {base_name}")
                
        logger.info(f"Found {len(valid_pairs)} matching audio-JAMS pairs")
        
        # Process files sequentially (more reliable than parallel for complex tasks)
        stats = {'total': len(valid_pairs), 'success': 0, 'error': 0}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process files in parallel
            futures = []
            for audio_path, jams_path in valid_pairs:
                futures.append(executor.submit(self.process_file, audio_path, jams_path))
                
            # Check results
            for future in tqdm(futures, desc="Processing files"):
                if future.result():
                    stats['success'] += 1
                else:
                    stats['error'] += 1
                    
        logger.info(f"Processing complete. Statistics:")
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully processed: {stats['success']}")
        logger.info(f"Errors: {stats['error']}")
        
        return stats
        
    def verify_output_integrity(self):
        """
        Verify that both output directories have matching files
        
        Returns:
        --------
        integrity : bool
            True if both directories have matching files
        """
        # Get all CQT image directories
        cqt_dirs = [d for d in self.cqt_output_dir.iterdir() if d.is_dir()]
        
        # Get all tablature directories
        tab_dirs = [d for d in self.tablature_output_dir.iterdir() if d.is_dir()]
        
        # Check if the same number of directories
        if len(cqt_dirs) != len(tab_dirs):
            logger.error(f"Number of directories doesn't match: CQT={len(cqt_dirs)}, Tab={len(tab_dirs)}")
            return False
            
        # Convert to sets for comparison
        cqt_dir_names = {d.name for d in cqt_dirs}
        tab_dir_names = {d.name for d in tab_dirs}
        
        # Check if all directories match
        if cqt_dir_names != tab_dir_names:
            logger.error("Directory names don't match between CQT and tablature outputs")
            return False
            
        # Check file counts within each directory
        mismatches = []
        
        for dir_name in cqt_dir_names:
            cqt_files = list((self.cqt_output_dir / dir_name).glob("*.png"))
            tab_files = list((self.tablature_output_dir / dir_name).glob("*.npy"))
            
            if len(cqt_files) != len(tab_files):
                mismatches.append((dir_name, len(cqt_files), len(tab_files)))
                
        if mismatches:
            logger.error(f"Found {len(mismatches)} directories with mismatched file counts:")
            for dir_name, cqt_count, tab_count in mismatches:
                logger.error(f"  {dir_name}: CQT={cqt_count}, Tab={tab_count}")
            return False
            
        # All checks passed
        logger.info("Integrity verification passed! Both directories have matching files.")
        return True

def main():
    parser = argparse.ArgumentParser(description='Generate CQT images and tablature data from audio and JAMS files')
    parser.add_argument('--audio_dir', required=True, help='Directory containing WAV audio files')
    parser.add_argument('--jams_dir', required=True, help='Directory containing JAMS annotation files')
    parser.add_argument('--cqt_output_dir', required=True, help='Directory to save CQT images')
    parser.add_argument('--tablature_output_dir', required=True, help='Directory to save tablature data')
    parser.add_argument('--segment_duration', type=float, default=0.2, help='Duration of each segment in seconds')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of parallel processes')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = GuitarCQTAndTablatureGenerator(
        audio_dir=args.audio_dir,
        jams_dir=args.jams_dir,
        cqt_output_dir=args.cqt_output_dir,
        tablature_output_dir=args.tablature_output_dir,
        segment_duration=args.segment_duration
    )
    
    # Process all files
    generator.process_all_files(max_workers=args.max_workers)
    
    # Verify output integrity
    generator.verify_output_integrity()

if __name__ == "__main__":
    main()
