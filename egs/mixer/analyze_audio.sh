#!/bin/bash

json_path=$1
output_file="analysis_output.txt"

echo "Analyzing audio files from $json_path" > $output_file

# Extract wav paths
python3 -c "
import json
with open('$json_path') as f:
    data = json.load(f)
for item in data['data']:
    print(item['wav'])
" | while read wav_path; do
    if [ ! -f "$wav_path" ]; then
        echo "Warning: $wav_path not found" >> $output_file
        continue
    fi

    # Get duration
    duration=$(ffprobe -v quiet -print_format json -show_format "$wav_path" | jq -r '.format.duration')

    # Get RMS (approximate using volumedetect)
    rms_info=$(ffmpeg -i "$wav_path" -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume")
    mean_volume=$(echo "$rms_info" | grep mean_volume | sed 's/.*mean_volume: \([0-9.-]*\) dB/\1/')

    echo "$wav_path: duration=$duration, mean_volume=$mean_volume" >> $output_file
done

echo "Analysis complete. See $output_file"