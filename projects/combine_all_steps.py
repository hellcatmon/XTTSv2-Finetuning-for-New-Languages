# Load aligned segments from JSON file
aligned_segments_path = '/content/downloads/audio_chunks/filenames.json'
with open(aligned_segments_path, 'r', encoding='utf-8') as f:
    aligned_segments = json.load(f)

# Combine all audio files with appropriate silences
combined_audio = AudioSegment.empty()
current_position = 0  # Track position in milliseconds

# Create silence segment for gaps between speech
for i, file_path in enumerate(output_files):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Get timing from original segment
        # Convert to ms
        segment_start = aligned_segments[i]["start"] * 1000

        # Add silence if needed before speech
        if segment_start > current_position:
            silence_duration = segment_start - current_position
            silence = AudioSegment.silent(duration=silence_duration)
            combined_audio += silence

        # Add the speech segment
        speech_segment = AudioSegment.from_file(file_path)
        combined_audio += speech_segment

        # Update position to end of current segment
        current_position = aligned_segments[i]["end"] * 1000
    else:
        print(f"[WARNING] File {
              file_path} is corrupted or empty and will be skipped.")

# Export final audio
final_output_path = os.path.join(output_dir, "final_tts_output.wav")
combined_audio.export(final_output_path, format="wav")
print(f"[INFO] Final TTS file saved: {final_output_path}")
