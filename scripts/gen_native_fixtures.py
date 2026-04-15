import json
import glob
import os
import shutil

# First, create native outputs
for path in glob.glob("tests/fixtures/golden/*_output.json"):
    if "native" in path:
        continue
    base = path.replace("_output.json", "")
    with open(path) as f:
        data = json.load(f)
    
    if "whisper_cpp" in base:
        native = {
            "engine": "whisper.cpp-native",
            "schema_version": "native-pilot-v1",
            "in_process": True,
            "segments": data.get("segments", []),
            "language": data.get("language", "en"),
            "text": data.get("text", "")
        }
        for s in native["segments"]:
            if "avg_logprob" in s: del s["avg_logprob"]
            if "no_speech_prob" in s: del s["no_speech_prob"]
            s["confidence"] = 0.95
    elif "insanely_fast" in base:
        native = {
            "engine": "insanely-fast-native",
            "schema_version": "native-pilot-v1",
            "in_process": True,
            "chunks": data.get("chunks", []),
            "text": data.get("text", "")
        }
    else:
        native = data

    with open(f"{base}_native_output.json", "w") as f:
        json.dump(native, f, indent=2)

for path in glob.glob("tests/fixtures/golden/*_output.srt"):
    if "native" in path:
        continue
    base = path.replace("_output.srt", "")
    shutil.copy(path, f"{base}_native_output.srt")

# Now update the corpus
for path in glob.glob("tests/fixtures/conformance/corpus/*.json"):
    with open(path) as f:
        data = json.load(f)
    
    new_engines = []
    for e in data["engines"]:
        new_engines.append(e)
        if "bridge" in e["engine"]:
            ne = e.copy()
            ne["engine"] = e["engine"].replace("bridge", "native")
            ne["artifact"] = e["artifact"].replace("_output", "_native_output")
            new_engines.append(ne)
            
    data["engines"] = new_engines
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
