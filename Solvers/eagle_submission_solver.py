import requests
import numpy as np
from Solvers.LSBSteg import decode

# api_base_url = "http://3.70.97.142:5000"
api_base_url = "http://localhost:3005"
# team_id="Lu2xdzj"
team_id = "xxx"


def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    payload_sent = {
        'teamId': team_id
    }
    response = requests.post(api_base_url+"/eagle/start", json=payload_sent)
    print(response)
    if response.status_code == 200 or response.status_code == 201:
        print("Game started successfully")
        data = response.json()
        footprints = data['footprint']
        return footprints
    else:
        print("error: ", response.status_code)


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
    '2': spect,
    '3': spect
    }
    for i in footpring:
        call model (footprint)
        if return == 1 return channel id
    
    return 0
    '''
    pass
  
def skip_msg(team_id):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    payload_sent = {
        'teamId': team_id
    }
    response = requests.post(api_base_url+"/eagle/skip-message", json=payload_sent)
    print(response)
    if response.status_code == 200 or response.status_code == 201:
        print("Message Skipped")
        if response == "End of message reached":
            return 0
        
        data = response.json()
        footprints = data['nextFootprint']
        return footprints
    else:
        print("error: ", response.status_code)
        
  
def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    payload_sent = {
        'teamId': team_id,
        'channelId': channel_id
    }
    response = requests.post(api_base_url+"/eagle/request-message", json=payload_sent)
    print(response)
    if response.status_code == 200 or response.status_code == 201:
        print("Message Request")
        data = response.json()
        encodedMsg = data['encodedMsg']
        return encodedMsg
    else:
        print("error: ", response.status_code)

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
    response = requests.post(api_base_url+"/eagle/submit-message", json=payload_sent)
    print(response)
    if response.status_code == 200 or response.status_code == 201:
        print("Message Submit")
        if response == "End of message reached":
            return 0
        
        data = response.json()
        footprints = data['nextFootprint']
        return footprints
    else:
        print("error: ", response.status_code)

    
  
def end_eagle(team_id):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    payload_sent = {
        'teamId': team_id,
    }
    response = requests.post(api_base_url+"/eagle/end-game", json=payload_sent)
    print(response)
    if response.status_code == 200 or response.status_code == 201:
        print("Game Ends")
        return response
    else:
        print("error: ", response.status_code)


def submit_eagle_attempt(team_id):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    footprints = init_eagle(team_id)
    print(footprints)
    while 1:
        if footprints == 0:
            end_eagle(team_id)
            return 0
        
        channel_id = select_channel(footprints)
        if channel_id == 0:
            footprints = skip_msg(team_id)
            continue
        else:
            image_msg = request_msg(team_id, channel_id)
            image_array = np.array(image_msg)
            text = decode(image_array)
            footprints = submit_msg(team_id, text)
            continue

submit_eagle_attempt(team_id)
