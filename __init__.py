import os
import csv
import time
import random
import spotipy
from typing import Dict, List, Optional
from collections import deque
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from development.course_detection.course_detection import CourseDetector
from development.state_detection.state_detector import StateDetector
from development.character_detection.character_detector import CharacterDetector
from development.vehicle_detection.vehicle_detector import VehicleDetector
from development.placement_detection.placement_detector import PlacementDetector


class GPINFO():
    def __init__(self):
        self.main_state = 0
        self.course_state = 0
        self.characters = ['Baby Mario','Baby Mario','Baby Mario','Baby Mario']
        self.vehicles = ['Bit Bike', 'Bit Bike', 'Bit Bike', 'Bit Bike']
        self.characterstats = self.csv_todict(file = 'nextgenstats/stats/characterstats.csv')
        self.vehiclestats = self.csv_todict(file = 'nextgenstats/stats/vehiclestats.csv')

    def csv_todict(self, file):
        with open(file) as f:
            dictionary = {row[0]: list(map(int, row[1:])) for row in csv.reader(f)}
        return dictionary

class ModelStore():
    def __init__(self, device = 'cpu'):
        self.models = {}
        self.device = device
        self.load_models()

    def load_models(self):
        flg_path = 'development/course_detection/models/flag_detector_20250221_122742.pth'
        state_path = 'production/models/menu_detection.pth'
        character_path = 'production/models/character_classifier.pth'
        vehicle_path = 'production/models/vehicle_classifier.pth'
        placement_path = 'production/models/placement_detection.pth'
        self.models['course_detector'] = CourseDetector(flag_model_path = flg_path, device = self.device)
        self.models['state_detector'] = StateDetector(model_path = state_path, device = self.device)
        self.models['character_detector'] = CharacterDetector(model_path = character_path, device = self.device)
        self.models['vehicle_detector'] = VehicleDetector(model_path=vehicle_path, device=self.device)
        self.models['placement_detector'] = PlacementDetector(model_path=placement_path, device=self.device)

class Song():
    def __init__(self,song_name,uri,img):
        self.song_name = song_name
        self.uri = uri
        self.img = img


class Course():
    def __init__(self, course_name = None, song_queue = None):
        self.course_name = course_name
        self.song_queue = song_queue

class SpotifyPlayer():
    def __init__(self):
        self.spotify = None
        self.playlist = {}
        self.songkey_dict = {}
        self.support_volume = False
        self.course_queued = None
        self.is_paused = False
        self.spotify_setup()

    def pause(self):
        if self.spotify.current_playback()['is_playing']:
            self.spotify.pause_playback(device_id=None)
            self.is_paused = True

    def resume(self):
        if not self.spotify.current_playback()['is_playing']:
            self.spotify.start_playback(device_id=None)
            self.is_paused = False

    def min_volume(self):
        if self.support_volume:
            self.spotify.volume(volume_percent = 0, device_id = None)

    def max_volume(self):
        if self.support_volume:
            self.spotify.volume(volume_percent = 100, device_id = None)

    def skip_tosong(self, song_uri: str):
        user_queue = self.spotify.queue().get('queue', [])
        song_names = [element['name'] for element in user_queue if 'name' in element]
        song_uris = [element['uri'] for element in user_queue if 'name' in element]
        self.min_volume()
        user_queue = self.spotify.queue()['queue']
        time.sleep(0.05)
        abort_count = 0
        for element in user_queue:
            self.spotify.next_track()
            if song_uri == element['uri']:
                self.max_volume()
                break
            abort_count += 1
            if abort_count == 3:
                break # prevent infinit skipping - Spotify API slow
        self.max_volume()

    def queue_songs(self, songs: List[Song]):
        for song in songs:
            self.spotify.add_to_queue(uri = song.uri, device_id = None)
        self.skip_tosong(song_uri = songs[0].uri)

    def search(self, search_query: str) -> Song:
        search_results = self.spotify.search(search_query, 1, 0, "track")
        tracks_items = search_results['tracks']['items']
        song_uri, song_name = tracks_items[0]['uri'], tracks_items[0]['name']
        image_url = tracks_items[0]['album']['images'][0]['url']
        search_song = Song(song_name=song_name, uri=song_uri, img=image_url)
        return search_song

    def get_song(self, song: str) -> Song:
        if song in self.songkey_dict:
            return self.songkey_dict[song]
        return self.search(song)

    def queue_newsong(self, course_name: str):
        song = self.playlist[course_name].song_queue.popleft()
        next_song = self.playlist[course_name].song_queue.popleft()
        self.queue_songs(songs = [self.get_song(next_song), self.get_song(song)])
        self.course_queued = course_name
        self.playlist[self.course_queued].song_queue.append(song)
        self.playlist[self.course_queued].song_queue.appendleft(next_song)

    def make_coursedict(self, file: str) -> Dict:
        course_dict = dict()
        f = open(file, 'r')
        datalines = f.readlines()
        for i in range(1, len(datalines)):
            data = datalines[i].split(',')
            course_name = data[0]
            data[-1] = data[-1].strip()
            course_songs = data[1:]
            random.shuffle(course_songs)
            q = deque()
            for j in range(len(course_songs)):
                q.append(course_songs[j])
            course_dict[course_name] = Course(course_name = course_name, song_queue = q)
        f.close()
        return course_dict

    def make_songkeydict(self, file: str) -> Dict:
        songkey_dict = dict()
        f = open(file, 'r')
        datalines = f.readlines()
        for i in range(1, len(datalines)):
            data = datalines[i].split(',')
            song_name = data[0].strip()
            song_uri = data[1].strip()
            song_img = data[2].strip()
            song = Song(song_name=song_name, uri=song_uri, img=song_img)
            songkey_dict[song_name] = song
        f.close()
        return songkey_dict

    def spotify_safetycheck(self):
        warning = False
        while self.spotify.current_playback()==None:
            if not warning:
                print('Activate Spotify')
                warning = True
        print('Spotify Connection Successful')

    def spotify_setup(self):
        file_path = 'credentials.txt'
        file = open(file_path,'r')
        cred_dict = {'username': 'None', 'client_id': 'None', 'client_secret': 'None', 'redirect_uri': 'None'}

        for cred in file.readlines():
            [label, key] = cred.split(' ')
            if label in cred_dict:
                cred_dict[label] = key.strip()
        scope = 'user-modify-playback-state user-read-playback-state'
        self.spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope,client_id=cred_dict['client_id'],
                                    client_secret=cred_dict['client_secret'],redirect_uri = cred_dict['redirect_uri'],
                                    username=cred_dict['username']))

        songkey_dict = self.make_songkeydict(file = 'audio/song_uri.csv')
        course_playlistfile = 'audio/playlists/rock_noon.csv'
        course_dict = self.make_coursedict(file = course_playlistfile)
        self.playlist = course_dict
        self.songkey_dict = songkey_dict
        self.spotify_safetycheck()
        self.support_volume = self.spotify.current_playback()['device']['supports_volume']

