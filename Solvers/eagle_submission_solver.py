import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import joblib
import numpy as np
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import sys

sys.stdout.flush()

class SteganographyException(Exception):
    pass

class LSBSteg():
    def __init__(self, im):
        self.image = im
        self.height, self.width, self.nbchannels = im.shape
        self.size = self.width * self.height

        self.maskONEValues = [1,2,4,8,16,32,64,128]
        #Mask used to put one ex:1->00000001, 2->00000010 .. associated with OR bitwise
        self.maskONE = self.maskONEValues.pop(0) #Will be used to do bitwise operations

        self.maskZEROValues = [254,253,251,247,239,223,191,127]
        #Mak used to put zero ex:254->11111110, 253->11111101 .. associated with AND bitwise
        self.maskZERO = self.maskZEROValues.pop(0)

        self.curwidth = 0  # Current width position
        self.curheight = 0 # Current height position
        self.curchan = 0   # Current channel position

    def put_binary_value(self, bits): #Put the bits in the image
        for c in bits:
            val = list(self.image[self.curheight,self.curwidth]) #Get the pixel value as a list
            if int(c) == 1:
                val[self.curchan] = int(val[self.curchan]) | self.maskONE #OR with maskONE
            else:
                val[self.curchan] = int(val[self.curchan]) & self.maskZERO #AND with maskZERO
            self.image[self.curheight,self.curwidth] = tuple(val)
            self.next_slot2() #Move "cursor" to the next space

    def next_slot(self):#Move to the next slot were information can be taken or put
        if self.curchan == self.nbchannels-1: #Next Space is the following channel
            self.curchan = 0
            if self.curwidth == self.width-1: #Or the first channel of the next pixel of the same line
                self.curwidth = 0
                if self.curheight == self.height-1:#Or the first channel of the first pixel of the next line
                    self.curheight = 0
                    if self.maskONE == 128: #Mask 1000000, so the last mask
                        raise SteganographyException("No available slot remaining (image filled)")
                    else: #Or instead of using the first bit start using the second and so on..
                        self.maskONE = self.maskONEValues.pop(0)
                        self.maskZERO = self.maskZEROValues.pop(0)
                else:
                    self.curheight +=1
            else:
                self.curwidth +=1
        else:
            self.curchan +=1

    def next_slot2(self):
        self.curchan += 1  # Move to the next channel
        if self.curchan == self.nbchannels:  # If all channels are exhausted
            self.curchan = 0  # Reset channel to the first one
            self.curwidth += 1  # Move to the next pixel on the same line
            if self.curwidth == self.width:  # If all pixels in the row are exhausted
                self.curwidth = 0  # Reset pixel position to the beginning of the row
                self.curheight += 1  # Move to the next row
                if self.curheight == self.height:  # If all rows are exhausted
                    self.curheight = 0  # Reset row position to the beginning
                    if self.maskONE == 128:  # If the last mask is reached
                        raise SteganographyException("No available slot remaining (image filled)")
                    else:  # Use the next available mask
                        self.maskONE = self.maskONEValues.pop(0)
                        self.maskZERO = self.maskZEROValues.pop(0)

    def read_bit(self): #Read a single bit int the image
        val = self.image[self.curheight,self.curwidth][self.curchan]
        # why & self.maskONE
        # val = 0b1011100
        # maskONE = 0b1000000
        # val & maskONE = 0b1000000
        val = int(val) & self.maskONE
        self.next_slot2()
        if val > 0:
            return "1"
        else:
            return "0"

    def read_byte(self):
        return self.read_bits(8)

    def read_bits(self, nb): #Read the given number of bits
        bits = ""
        for i in range(nb):
            bits += self.read_bit()
        return bits

    def byteValue(self, val):
        return self.binary_value(val, 8)


    # test case:
    # # Expected binary value: '0110'
    # Explanation: Binary representation of 6 with bitsize 4.
    #binary_value(6, 4) == '0110'
    ###
    def binary_value(self, val, bitsize): #Return the binary value of an int as a byte
        binval = bin(val)[2:] ## delete 0b
        if len(binval) > bitsize:
            raise SteganographyException("binary value larger than the expected size")
        while len(binval) < bitsize:
            binval = "0"+binval
        return binval

    ## encode text by steps:
    # 1. put len of text
    # 2. put each char of text
    def encode_text(self, txt):
        l = len(txt)
        binl = self.binary_value(l, 16) #Length coded on 2 bytes so the text size can be up to 65536 bytes long
        self.put_binary_value(binl) #Put text length coded on 4 bytes
        for char in txt: #And put all the chars
            c = ord(char)
            self.put_binary_value(self.byteValue(c))
        return self.image

    # 1. read len of text
    # 2. read each char of text
    # 3. return the text
    def decode_text(self):
        ls = self.read_bits(16) #Read the text size in bytes
        l = int(ls,2)
        i = 0
        unhideTxt = ""
        while i < l: #Read all bytes of the text
            tmp = self.read_byte() #So one byte
            i += 1
            unhideTxt += chr(int(tmp,2)) #Every chars concatenated to str
        return unhideTxt

    def encode_image(self, imtohide):
        w = imtohide.width
        h = imtohide.height
        if self.width*self.height*self.nbchannels < w*h*imtohide.channels:
            raise SteganographyException("Carrier image not big enough to hold all the datas to steganography")
        binw = self.binary_value(w, 16) #Width coded on to byte so width up to 65536
        binh = self.binary_value(h, 16)
        self.put_binary_value(binw) #Put width
        self.put_binary_value(binh) #Put height
        for h in range(imtohide.height): #Iterate the hole image to put every pixel values
            for w in range(imtohide.width):
                for chan in range(imtohide.channels):
                    val = imtohide[h,w][chan]
                    self.put_binary_value(self.byteValue(int(val)))
        return self.image


    def decode_image(self):
        width = int(self.read_bits(16),2) #Read 16bits and convert it in int
        height = int(self.read_bits(16),2)
        unhideimg = np.zeros((width,height, 3), np.uint8) #Create an image in which we will put all the pixels read
        for h in range(height):
            for w in range(width):
                for chan in range(unhideimg.channels):
                    val = list(unhideimg[h,w])
                    val[chan] = int(self.read_byte(),2) #Read the value
                    unhideimg[h,w] = tuple(val)
        return unhideimg

    def encode_binary(self, data):
        l = len(data)
        if self.width*self.height*self.nbchannels < l+64:
            raise SteganographyException("Carrier image not big enough to hold all the datas to steganography")
        self.put_binary_value(self.binary_value(l, 64))
        for byte in data:
            byte = byte if isinstance(byte, int) else ord(byte) # Compat py2/py3
            self.put_binary_value(self.byteValue(byte))
        return self.image

    def decode_binary(self):
        l = int(self.read_bits(64), 2)
        output = b""
        for i in range(l):
            output += chr(int(self.read_byte(),2)).encode("utf-8")
        return output

def encode(image: np.ndarray, message: str) -> np.array:
    steg = LSBSteg(image)
    img_encoded = steg.encode_text(message)
    return img_encoded

def decode(encoded: np.array) -> str:
    steg = LSBSteg(encoded)
    return steg.decode_text()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import warnings
# Disabling Future Warnings0
warnings.filterwarnings(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.models import load_model as load
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import numpy as np

import matplotlib.pyplot as plt

model = tf.keras.models.load_model('with_30epochs.h5')

fake = np.load('fake.npz')
real = np.load('real.npz')
specto = np.concatenate((fake['x'], real['x']), axis=0).astype(np.float64)  # Change dtype to float32
labels = np.concatenate(([0 for i in range(750)], [1 for i in range(750)]), axis=0)

clip_min = -1e10
clip_max = 1e10
specto = np.clip(specto, clip_min, clip_max)

mean = np.mean(specto, axis=(0, 1))  # Compute mean across samples and time steps
std = np.std(specto, axis=(0, 1))    # Compute standard deviation across samples and time steps

import requests
import numpy as np

api_base_url = "http://3.70.97.142:5000"
# api_base_url = "http://localhost:3005"
team_id="Lu2xdzj"
# team_id = "xxx"

session = requests.Session()

def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    payload_sent = {
        'teamId': team_id
    }
    response = session.post(api_base_url+"/eagle/start", json=payload_sent)
    if response.ok:
        data = response.json()
        footprints = data['footprint']
        return footprints
    
    print("error: ", response.status_code)
    return 0
  

def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.
    '''

    '''
    foorprint = {
    '1': spect,
    }
    for i in footpring:
        call model (footprint)
        if return == 1 return channel id

    return 0
    '''
    for channel, spect in footprint.items():
        spect = np.array(spect).astype(np.float64)
        spect = np.clip(spect, clip_min, clip_max)
        mean2 = np.mean(spect, axis=(0, 1))

        if abs(mean2) < 0.5:
          continue

        normalized_spect = (spect - mean) / std
        
        spect = np.expand_dims(normalized_spect, axis=0)
        prediction = model.predict(spect)[0]

        if prediction > 0.5:
            return int(channel)

    return 0

def skip_msg(team_id):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    payload_sent = {
        'teamId': team_id
    }
    response = session.post(api_base_url+"/eagle/skip-message", json=payload_sent)
    if response.ok:
        data = response.json()
        footprints = data['nextFootprint']
        return footprints
    else:
        if response.text == "End of message reached":
          print(response.text, "status: ", response.status_code)
        else:
          print("error: ", response.status_code)
    return 0

def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    payload_sent = {
        'teamId': team_id,
        'channelId': channel_id
    }
    response = session.post(api_base_url+"/eagle/request-message", json=payload_sent)
    if response.ok:
        data = response.json()
        encodedMsg = data['encodedMsg']
        return encodedMsg
    
    print("error: ", response.status_code)
    return 0

def submit_msg(team_id, decoded_msg):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    payload_sent = {
        'teamId': team_id,
        'decodedMsg': decoded_msg
    }
    response = session.post(api_base_url+"/eagle/submit-message", json=payload_sent)
    if response.ok:
        data = response.json()
        footprints = data['nextFootprint']
        return footprints
    else:
        if response.text == "End of message reached":
          print(response.text, "status: ", response.status_code)
        else:
          print("error: ", response.status_code)
    return 0


def end_eagle(team_id):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    payload_sent = {
        'teamId': team_id,
    }
    response = session.post(api_base_url+"/eagle/end-game", json=payload_sent)
    if response.ok:
        print("Game Ends ", response.text)
    else:
        print("error: ", response.status_code)
    return 0



import time

start_time = time.time()

footprints = init_eagle(team_id)
while 1:
  if footprints == 0:
      end_eagle(team_id)
      break

  channel_id = select_channel(footprints)
  if channel_id == 0:
      footprints = skip_msg(team_id)
      continue
  else:
      image_msg = request_msg(team_id, channel_id)
      image_array = np.array(image_msg, dtype=np.uint8)
      text = decode(image_array)
      footprints = submit_msg(team_id, text)
      continue

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")