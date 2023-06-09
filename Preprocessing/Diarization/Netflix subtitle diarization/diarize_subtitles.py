from pyannote.audio import Pipeline
from lxml import etree
import pandas as pd
import json

with open('secrets.json', 'r') as f:
    secrets = json.load(f)
HUGGING_FACE_APIKEY = secrets["HUGGING_FACE_APIKEY"]

audio_path = "D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/TheOfficeSeries/Episodes/1/S01E01.mp3"
transcription_path = "D:/OneDrive - Universiteit Utrecht/Documents/000 - Documenti/PROJECTS/PSYCHE/TheOfficeSeries/Subtitles/S01/S01E1.txt"

# Load and parse the XML file
tree = etree.parse(transcription_path)
root = tree.getroot()

# Extract the subtitles
subtitles = []
for p in root.iter("{*}p"):
    start = int(p.get("begin")[:-1])
    end = int(p.get("end")[:-1])
    text = " ".join(p.itertext())
    subtitles.append({"start": start, "end": end, "text": text})

df_subs = pd.DataFrame(subtitles)

# Initialize pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_APIKEY)

# Apply the pipeline to the audio file
diarization = pipeline(audio_path)

# Map the speakers to the subtitles
df_subs['start'] /= 1e9  # Convert start time to seconds
df_subs['end'] /= 1e9  # Convert end time to seconds

df_subs['speaker'] = 'Unknown'
for i, row in df_subs.iterrows():
    speaker = diarization.get_label_at(row['start'])
    if speaker is not None:
        df_subs.at[i, 'speaker'] = speaker

print(df_subs)
