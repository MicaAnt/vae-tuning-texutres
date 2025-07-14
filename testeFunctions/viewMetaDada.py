import csv
import sys

def show_metadata(csv_path="../midiDataTest/commu_meta.csv", track_id="commu00002"):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if r.get("id") == track_id), None)

    if row is None:
        print(f"ID '{track_id}' não encontrado.")
    else:
        print(f"📄 Metadados para '{track_id}':\n")
        for key, value in row.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Permite passar um ID como argumento opcional
    track_id = sys.argv[1] if len(sys.argv) > 1 else "commu00001"
    show_metadata(track_id=track_id)