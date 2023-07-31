import json
import os

base_jase = {
    "name": "acoustic_grand_piano",
    "minPitch": 21,
    "maxPitch": 108,
    "durationSeconds": 3.0,
    "releaseSeconds": 1.0,
    "percussive": False,
    "velocities": [85]}

if __name__ == '__main__':

    for d in os.walk('.'):
        d = d[0]
        base_jase['name'] = d
        json.dump(base_jase, open(d + '/instrument.json', 'w+'))

