from __init__ import *
import os
from pathlib import Path


script_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
file_path = os.path.join(script_dir, "credentials.txt")

def spotify_setup():
    file = open(file_path, 'r')
    cred_dict = {'username': 'None', 'client_id': 'None', 'client_secret': 'None', 'redirect_uri': 'None'}
    for cred in file.readlines():
        [label, key] = cred.split(' ')
        if label in cred_dict:
            cred_dict[label] = key.strip()
    scope = 'user-modify-playback-state user-read-playback-state'
    spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=cred_dict['client_id'],
                                                             client_secret=cred_dict['client_secret'],
                                                             redirect_uri=cred_dict['redirect_uri'],
                                                             username=cred_dict['username']))
    return spotify

def get_songinfo():
    spotify = spotify_setup()
    current_track = spotify.current_playback()
    track_name = current_track["item"]["name"]
    track_uri = current_track["item"]["uri"]
    album_image = current_track["item"]["album"]["images"][0]["url"]
    print(f"🎵 Now Playing: {track_name}")
    print(f"🔗 URI: {track_uri}")
    print(f"🖼️ Album Image: {album_image}")
    print(f'{track_name},{track_uri},{album_image}')

get_songinfo()