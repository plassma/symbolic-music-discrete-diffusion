import requests
import json
import os
from multiprocessing import Pool


def download(arg):
    instrument, path, sn = arg
    if not os.path.exists(path):
        s = requests.get(url + instrument + sn).content
        with open(path, 'wb') as f:
            f.write(s)
        print("downloaded", path)

# cache sgm samples for offline use
if __name__ == '__main__':
    url = 'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus/'

    os.makedirs('sgm_plus', exist_ok=True)
    file = requests.get(url + 'soundfont.json').content
    instruments = json.loads(file)

    with open('sgm_plus/soundfont.json', 'w') as f:
        f.write(file.decode('utf-8'))

    pool = Pool(40)

    for instrument in instruments['instruments'].values():
        inst_dir = 'sgm_plus/' + instrument
        os.makedirs(inst_dir, exist_ok=True)
        inst_json = requests.get(url + instrument + '/instrument.json').content
        instrument_json = json.loads(inst_json)

        with open('sgm_plus/' + instrument + '/instrument.json', 'w') as f:
            f.write(file.decode('utf-8'))

        for v in instrument_json['velocities']:
            args = []
            for p in range(instrument_json['minPitch'], instrument_json['maxPitch']):
                sn = f'/p{p}_v{v}.mp3'
                path = 'sgm_plus/' + instrument + sn
                args.append((instrument, path, sn))
            pool.map(download, args)

