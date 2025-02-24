import random
import time
from collections import deque
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from development.course_detector import CourseDetector

class ModelStore():
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        self.models['course_detector'] = CourseDetector(flag_model_path='development/course_detection/models/flag_detector_20250221_122742.pth')


class SpotifyPlayer():
    def __init__(self):
        self.spotify = None
        self.playlist = {}
        self.songkey_dict = {}
        self.support_volume = False
        self.course_queued = None
        self.spotify_setup()

    def min_volume(self):
        if self.support_volume:
            self.spotify.volume(volume_percent = 0, device_id = None)

    def max_volume(self):
        if self.support_volume:
            self.spotify.volume(volume_percent = 100, device_id = None)

    def skip_tosong(self, song_uri):
        self.min_volume()
        user_queue = self.spotify.queue()['queue']
        time.sleep(0.1)
        for element in user_queue:
            self.spotify.next_track()
            if song_uri == element['uri']:
                self.max_volume()
                break
        self.max_volume()

    def queue_songs(self,songs):
        for song in songs:
            self.spotify.add_to_queue(uri = song.uri, device_id = None)
        self.skip_tosong(song_uri = songs[0].uri)

    def search(self, searchQuery):
        search_results = self.spotify.search(searchQuery, 1, 0, "track")
        tracks_items = search_results['tracks']['items']
        song_uri, song_name = tracks_items[0]['uri'], tracks_items[0]['name']
        image_url = tracks_items[0]['album']['images'][0]['url']
        search_song = Song(song_name=song_name, uri=song_uri, img=image_url)
        return search_song

    def get_song(self, song):
        if song in self.songkey_dict:
            return self.songkey_dict[song]
        return self.search(song)

    def queue_newsong(self, course_name):
        song = self.playlist[course_name].song_queue.popleft()
        next_song = self.playlist[course_name].song_queue.popleft()
        self.queue_songs(songs = [self.get_song(song), self.get_song(next_song)])
        self.course_queued = course_name
        self.playlist[self.course_queued].song_queue.append(song)
        self.playlist[self.course_queued].song_queue.appendleft(next_song)

    def make_coursedict(self, file):
        course_dict = dict()
        f = open(file, 'r')
        datalines = f.readlines()
        for i in range(1, len(datalines)):
            data = datalines[i].split(',')
            course_name = data[0]
            data[-1] = data[-1].strip()
            course_songs = data[1:]
            #random.shuffle(course_songs)
            q = deque()
            for j in range(len(course_songs)):
                q.append(course_songs[j])
            course_dict[course_name] = Course(course_name=course_name, song_queue=q)
        f.close()
        return course_dict

    def make_songkeydict(self, file):
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
        file = open('credentials.txt','r')
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
        course_playlistfile = 'audio/playlists/rock.csv'
        course_dict = self.make_coursedict(file = course_playlistfile)
        self.playlist = course_dict
        self.songkey_dict = songkey_dict
        self.spotify_safetycheck()


class Song():
    def __init__(self,song_name,uri,img):
        self.song_name = song_name
        self.uri = uri
        self.img = img


class Course():
    def __init__(self, course_name = None, song_queue = None):
        self.course_name = course_name
        self.song_queue = song_queue
