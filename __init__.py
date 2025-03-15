import os
import csv
import time
import random
import shutil
import spotipy
from urllib.request import urlopen
from collections import deque
from typing import Dict, Deque, List, Optional
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from development.course_detection.course_detection import CourseDetector
from development.state_detection.state_detector import StateDetector
from development.character_detection.character_detector import CharacterDetector
from development.vehicle_detection.vehicle_detector import VehicleDetector
from development.placement_detection.placement_detector import PlacementDetector
from development.countdown_detection.countdown_detector import CountdownDetector


class Course():
    """Represents a course with an optional name and a song queue"""

    def __init__(self, course_name: str = None, song_queue: Deque = None):
        self.course_name = course_name
        self.song_queue = song_queue



class GPINFO():
    def __init__(self):
        """Initialize game state variables"""
        self.main_state = 0
        self.course_state = 0
        self.player_count = 4

        # Default player setup
        self.characters = ['Baby Mario'] * 4
        self.vehicles = ['Bit Bike'] * 4
        self.places = [12, 11, 10, 9]

        # Track course history and start time
        self.course_history = []
        self.course_start = None

        # Load character, vehicle, and course data from csv files
        self.characterstats = self.csv_todict(file = 'nextgenstats/stats/characterstats.csv')
        self.vehiclestats = self.csv_todict(file = 'nextgenstats/stats/vehiclestats.csv')
        self.course_data = self.csv_todict(file = 'nextgenstats/stats/coursedata.csv')

    @staticmethod
    def clear_directory(directory_path):
        """Deletes all files and subdirectories inside a directory."""
        if not os.path.exists(directory_path):
            return
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Delete file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete folder and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    @staticmethod
    def csv_todict(file_path):
        """Loads a CSV file into a dictionary, skipping the header row"""
        try:
            with open(file_path) as f:
                reader = csv.reader(f)
                next(reader)
                return  {row[0]: list(map(str, row[1:])) for row in reader}
        except FileNotFoundError:
            print(f'Error: {file_path} not found')
            return {}
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return {}



class ModelStore():
    """Handles loading and storing multiple ML models for game-related detections"""

    def __init__(self, device = 'cpu'):
        self.models = {}
        self.device = device
        self.load_models()

    def load_models(self):
        """Loads all required models and stores them in a dictionary"""
        model_paths = {
            'course_detector': 'development/course_detection/models/flag_detector_20250221_122742.pth',
            'state_detector': 'production/models/menu_detection.pth',
            'character_detector' : 'production/models/character_classifier.pth',
            'vehicle_detector': 'production/models/vehicle_classifier.pth',
            'countdown_detector': 'production/models/countdown_detection.pth',
            'placement_detector': 'production/models/placement_detection.pth'
        }
        # Initialize model objects
        self.models = {
            'course_detector': CourseDetector(flag_model_path = model_paths['course_detector'], device = self.device),
            'state_detector': StateDetector(model_path=model_paths['state_detector'], device=self.device),
            'character_detector': CharacterDetector(model_path=model_paths['character_detector'], device=self.device),
            'vehicle_detector': VehicleDetector(model_path=model_paths['vehicle_detector'], device=self.device),
            'countdown_detector': CountdownDetector(model_path=model_paths['countdown_detector'], device=self.device),
            'placement_detector': PlacementDetector(model_path=model_paths['placement_detector'], device=self.device),
        }



class Song():
    """Represents a song with name, Spotify URI, and image"""

    def __init__(self, song_name: str, uri: str, img: str):
        self.song_name = song_name
        self.uri = uri
        self.img = img



class SpotifyPlayer():
    """Handles Spotify playback, song queuing, and course-based playlist"""

    def __init__(self):
        self.spotify = None
        self.playlist = {}
        self.songkey_dict = {}
        self.course_queued = None
        self.song_queued = None
        self.song_img = None
        self.is_paused = False
        self.spotify_setup()

    def pause(self):
        """Pauses playback if music is currently playing"""
        playback = self.spotify.current_playback()
        if playback and not playback['is_playing']:
            self.spotify.pause_playback(device_id=None)
            self.is_paused = True

    def resume(self):
        """Resumes playback if music is currently paused"""
        playback = self.spotify.current_playback()
        if playback and not playback['is_playing']:
            self.spotify.start_playback(device_id=None)
            self.is_paused = False

    def set_volume(self, volume: int):
        playback = self.spotify.current_playback()
        if playback and playback['device']['supports_volume']:
            self.spotify.volume(volume_percent= volume, device_id=None)

    def skip_tosong(self, song_uri: str):
        """Skips to a specific song in the queue, limiting excessive skips"""
        user_queue = self.spotify.queue().get('queue', [])
        self.set_volume(0)
        time.sleep(0.05)

        abort_count = 0
        for element in user_queue:
            self.spotify.next_track()
            if song_uri == element['uri']:
                break
            abort_count += 1
            if abort_count == 3: # Prevents excessive skipping due to API latency
                break

        self.set_volume(100)

    def queue_songs(self, songs: List[Song]):
        """Adds songs to the queue and starts playing the first one"""
        for song in songs:
            self.spotify.add_to_queue(uri = song.uri, device_id = None)
        self.skip_tosong(song_uri = songs[0].uri)

    def search(self, search_query: str) -> Song:
        """Searches for a song on Spotify and returns its details in Song Class"""
        search_results = self.spotify.search(search_query, 1, 0, "track")
        track_items = search_results.get('tracks', {}).get('items', [])

        if not track_items:
            raise ValueError(f'No results found for: {search_query}')

        track = track_items[0]
        search_song = Song(
            song_name = track['name'],
            uri = track['uri'],
            img = track['album']['images'][0]['url'],
        )
        return search_song

    def get_song(self, song_name: str) -> Song:
        """Retrieves a song from the dictionary or searches for it on Spotify"""
        return self.songkey_dict.get(song_name) or self.search(song_name)

    def queue_newsong(self, course_name: str):
        """Queues the next song for the given course and updates tracking variables"""
        course = self.playlist.get(course_name)
        if not course or not course.song_queue:
            return

        song_name = course.song_queue.popleft()
        next_song_name = course.song_queue.popleft()

        song = self.get_song(song_name)
        next_song = self.get_song(next_song_name)

        self.queue_songs(songs = [song, next_song])

        self.course_queued = course_name
        self.playlist[self.course_queued].song_queue.append(song)
        self.playlist[self.course_queued].song_queue.appendleft(next_song)

        self.song_queued = self.songkey_dict.get(song.song_name)
        if self.song_queued:
            self.song_img = urlopen(self.song_queued.img).read()

    def load_csv_to_dict(self, file_path: str) -> Dict[str, List[str]]:
        """Loads CSV data into a dictionary"""
        with open(file_path, 'r') as file:
            lines = [line.strip().split(",") for line in file.readlines()[1:]]
        return {data[0]: data[1:] for data in lines}

    def make_coursedict(self, file_path: str) -> Dict:
        """Creates a dictionary of courses mapped to their shuffled song queues"""
        course_dict = dict()
        data = self.load_csv_to_dict(file_path = file_path)

        for course_name, songs in data.items():
            random.shuffle(songs)
            course_dict[course_name] = Course(course_name = course_name, song_queue = deque(songs))

        return course_dict

    def make_songkeydict(self, file_path: str) -> Dict[str, Song]:
        """Creates a dictionary mapping song names to Song objects"""
        songkey_dict = dict()
        data = self.load_csv_to_dict(file_path)

        for song_name, song_info in data.items():
            song = Song(song_name = song_name, uri = song_info[0].strip(), img = song_info[1].strip())
            songkey_dict[song_name] = song

        return songkey_dict

    def spotify_safetycheck(self):
        """Waits for Spotify connection to be active"""
        warning_displayed = False
        while not self.spotify.current_playback():
            if not warning_displayed:
                print('Waiting for Spotify to be activated...')
                warning_displayed = True
        print('Spotify Connection Successful')

    def load_credentials(self, file_path: str) -> Dict[str, str]:
        """Loads Spotify credentials from a file"""
        credentials = {"username": None, "client_id": None, "client_secret": None, "redirect_uri": None}

        with open(file_path, "r") as file:
            for line in file:
                key, value = line.strip().split(" ", 1)
                if key in credentials:
                    credentials[key] = value

        return credentials

    def spotify_setup(self):
        """Sets up Spotify API authentication and loads playlist"""
        credentials = self.load_credentials(file_path = "credentials.txt")
        scope = 'user-modify-playback-state user-read-playback-state'

        self.spotify = spotipy.Spotify(
            auth_manager = SpotifyOAuth(scope = scope,
                        client_id = credentials['client_id'],
                        client_secret = credentials['client_secret'],
                        redirect_uri = credentials['redirect_uri'],
                        username = credentials['username']
            )
        )

        self.songkey_dict = self.make_songkeydict(file_path = 'audio/song_uri.csv')
        self.playlist = self.make_coursedict(file_path = 'audio/playlists/rock_noon.csv' )
        self.spotify_safetycheck()