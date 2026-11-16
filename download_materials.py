#!/usr/bin/env python3
"""Download material ZIPs from an Anonymous GitHub repository using Python 3."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

FILES = [{'group': 'figure', 'name': 'Figure_videos_01.zip', 'bytes': 93642294, 'sha256': 'efa0c73d92de08871befb7283494ed180acaf8e0392b6a5d9f760684cb8344ec', 'files': 55}, {'group': 'figure', 'name': 'Figure_videos_02.zip', 'bytes': 92838499, 'sha256': '14269e0b726ab144fbf41416c2e9b89309c4b0d1c40787123ca60ea3e2fec2f6', 'files': 73}, {'group': 'figure', 'name': 'Figure_videos_03.zip', 'bytes': 91105989, 'sha256': 'f5cdd4088431b3574003dc4412ae851a811d51d67e380416fa308f72d2c73eff', 'files': 86}, {'group': 'composite', 'name': 'Composite180_01.zip', 'bytes': 84988029, 'sha256': 'cbfd7da0fec02c88bbe28cbd25c5c51d61b2bf86ede9f577034c05bd2c2dfc92', 'files': 68}, {'group': 'composite', 'name': 'Composite180_02.zip', 'bytes': 89205217, 'sha256': 'f768cc3eb17c46c0dac3ffafcfae2db0955169efab24e4767daab49e430a4caa', 'files': 60}, {'group': 'composite', 'name': 'Composite180_03.zip', 'bytes': 87247198, 'sha256': '082d9c56da71d87386ea02c36d21f1260da06ee13bbed8213e437639b7bb6c53', 'files': 70}, {'group': 'composite', 'name': 'Composite180_04.zip', 'bytes': 89142817, 'sha256': '61cfa3dbcabcc80f33e734b54e54e8b8127d41d5e3f17330322a0f0b71056a00', 'files': 50}, {'group': 'composite', 'name': 'Composite180_05.zip', 'bytes': 93502731, 'sha256': '9207a24f33b097210ea9be80e416d26cf2032576f672bdcc47c5b510bebf2fe2', 'files': 70}, {'group': 'composite', 'name': 'Composite180_06.zip', 'bytes': 93397001, 'sha256': '67a6acdb79b6c83789fbc271484038632b348454fb26df797edeef28727c76ee', 'files': 95}, {'group': 'composite', 'name': 'Composite180_07.zip', 'bytes': 94016358, 'sha256': '8cbea69acc3a0ad0af6b235579fd0ca51d6ed00770eed02387cf4ce82af675c0', 'files': 90}, {'group': 'composite', 'name': 'Composite180_08.zip', 'bytes': 93742123, 'sha256': 'cb07d842f915179458c1de2983bb8ed145463e6d1ed500e93307062f438fb319', 'files': 65}, {'group': 'composite', 'name': 'Composite180_09.zip', 'bytes': 93047944, 'sha256': 'af28a0ade7ef35cc885dd33e4afe2a8586dc132727c1218bf571ddb75768f1f6', 'files': 70}, {'group': 'composite', 'name': 'Composite180_10.zip', 'bytes': 91250238, 'sha256': '768812efb10057cfc64a52304bc2d601770522b7a7439d0caa9b503ae99097ab', 'files': 75}, {'group': 'composite', 'name': 'Composite180_11.zip', 'bytes': 89273973, 'sha256': '0faa038fc8a2f9f4d04d94cc7898da289041281b5a431d8d1da9b0b4a7e132f7', 'files': 75}, {'group': 'composite', 'name': 'Composite180_12.zip', 'bytes': 88995476, 'sha256': '3dfefe97653da8dacfab6dad75b8f72db3e707420fedebd6525b52f0ede56218', 'files': 45}, {'group': 'composite', 'name': 'Composite180_13.zip', 'bytes': 88131441, 'sha256': '45a5daa8e4cc3bc9de11e5d86e6f5192f048b6cc340e257a8bcf61c9c0568399', 'files': 75}, {'group': 'composite', 'name': 'Composite180_14.zip', 'bytes': 9289012, 'sha256': '8ec74a2f0f9bacffea3fd1fc46752cbd22901c31e02ce3ecee6f2a01200e2d78', 'files': 5}, {'group': 'baseline', 'name': 'S3_baselines_01.zip', 'bytes': 82696928, 'sha256': '0a024c35738f140e6f86e5770e2e498451f77110cbb589f4926b18d5b381b0f4', 'files': 55}, {'group': 'baseline', 'name': 'S3_baselines_02.zip', 'bytes': 65093330, 'sha256': '3b18da3cc8a5b4a2998f3509c6db61d1afaf6aedc5aeab86750169c09b6f8307', 'files': 20}, {'group': 'code', 'name': 'QuanTopo_code.zip', 'bytes': 112831, 'sha256': '6ffd1f021b0c4103046b0945ca68fbc31174284631a4ffcdb4c68546d3f64ddf', 'files': 91}]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", help="Anonymous repository URL, e.g. https://anonymous.4open.science/r/NAME-ID/")
    parser.add_argument("--group", choices=["all", "figure", "composite", "baseline", "code"], default="all")
    parser.add_argument("--output", default="QuanTopo_downloads")
    args = parser.parse_args()
    repository = args.repository or input("Anonymous repository URL: ").strip()
    parsed = urllib.parse.urlsplit(repository)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme != "https" or len(parts)<2 or parts[0]!="r":
        parser.error("Provide an HTTPS anonymous repository URL containing /r/REPOSITORY-ID/")
    base = f"https://{parsed.netloc}/api/repo/{urllib.parse.quote(parts[1], safe='')}/file/downloads/"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for item in FILES:
        if args.group!="all" and item["group"]!=args.group:
            continue
        target = out/item["name"]
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==item["sha256"]:
            print("Already downloaded:", target.name, flush=True)
            continue
        for attempt in range(3):
            partial = target.with_suffix(".zip.part")
            try:
                req = urllib.request.Request(base+urllib.parse.quote(item["name"])+"?download=1", headers={"User-Agent":"QuanTopo-materials-downloader/1.0"})
                digest = hashlib.sha256()
                with urllib.request.urlopen(req, timeout=180) as response, partial.open("wb") as stream:
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        digest.update(block)
                        stream.write(block)
                if partial.stat().st_size!=item["bytes"] or digest.hexdigest()!=item["sha256"]:
                    raise ValueError("Downloaded bytes failed integrity verification")
                os.replace(partial, target)
                print("Downloaded:", target.name, flush=True)
                break
            except Exception as exc:
                partial.unlink(missing_ok=True)
                if attempt==2:
                    raise SystemExit(f"Download failed for {target.name}: {exc}. Check that the anonymous snapshot is refreshed and available.")
                time.sleep(2*(attempt+1))
    print("Extract the downloaded ZIPs into the same destination folder. No joining or concatenation is required.")

if __name__=="__main__":
    main()
