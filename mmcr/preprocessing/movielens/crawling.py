import datetime as dt
import multiprocessing as mp
import os
import pandas as pd
from tqdm import tqdm
from yt_dlp import YoutubeDL
import argparse

audio_codec='flac'
audio_format='flac',
audio_sample_rate=44100
audio_bit_depth=16
video_codec='h264'
video_format='mp4'
video_mode='bestvideoaudio'
video_frame_rate=30

def _download(ytid, out_dir):
    ydl_opts = {
        "quiet": True,
        "outtmpl": out_dir,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        "external_downloader": "ffmpeg",
        "external_downloader_args": [
            '-f', video_format,
            '-r', str(video_frame_rate),
            '-vcodec', video_codec,
            '-acodec', 'aac',
            '-strict', 'experimental',
            "-http_proxy","socks5://127.0.0.1:1080"
            ]
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
    except KeyboardInterrupt:
        raise
    except Exception:
        pass

def main(args):
    os.makedirs(os.path.join(args.save_path), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "mp4"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "error"), exist_ok=True)
    df = pd.read_csv("../dataset/ml-20m/ml-youtube.csv")
    _yids = list(df['youtubeId'])
    already_down_ids = [i.replace(".mp4", "") for i in os.listdir(os.path.join(args.save_path, "mp4")) if ".mp4" in i]
    yids = list(set(_yids).difference(already_down_ids))
    output_paths = [os.path.join(args.save_path, "mp4", i + ".mp4") for i in _yids]
    print(len(yids), len(already_down_ids)) # 8481 17350 -> 8386 17446 -> 8379 17892
    import time
    time.sleep(5)
    with mp.Pool(processes=mp.cpu_count() - 5) as pool:
        pool.starmap(_download, zip(_yids, output_paths))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="../dataset/ml-20m/content")
    args = parser.parse_args()
    main(args)