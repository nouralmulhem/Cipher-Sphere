import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers
import cv2

api_base_url = "http://16.16.170.3/"
# team_id = Lu2xdzj (take care to use the same team id and start game 🧑🏼‍🚒)
team_id="xxx"
total_budget=0
def init_fox(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    # payload_sent = {
    #     'teamId': team_id
    # }
    # response = requests.post(api_base_url+"/fox/start", json=payload_sent)
    # if response.status_code == 200 or response.status_code == 201:
    #     data = response.json()
    #     msg = data['msg']
    #     carrier_image = data['carrier_image']
    # else:
    #     print("error: ", response.status_code)
    msg="hello"
    carrier_image=cv2.imread('./SteganoGAN/sample_example/encoded.png')
    return msg, carrier_image

def generate_message_array(message, image_carrier):  
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''
    def split_string_into_two_chars(input_string):
        pairs = [input_string[i:i+2] for i in range(0, len(input_string), 2)]
        if len(input_string) % 2 != 0:
            pairs.append(pairs.pop()[0])
        return pairs
    # new_message = split_string_into_two_chars(message)
    new_message =[message]
    index=0
    channel=0
    def sent_message(fake_msg,real_msg):
        global total_budget
        global channel
        message_entities=['E' for _ in range(3)]
        messages = [image_carrier for _ in range(3)]
        for i in range(len(fake_msg)):
            image=encode(image_carrier.copy(),fake_msg[i]).tolist()
            messages[channel]=image
            message_entities[channel]='F'
            channel=(channel+1)%3
            total_budget-=1
        
        for i in range(len(real_msg)):
            image=encode(image_carrier.copy(),real_msg[i]).tolist()
            messages[channel]=image
            message_entities[channel]='R'
            channel=(channel+1)%3

    while(total_budget>0):
        if(total_budget>=2):
            sent_message(["fake","fffff"],[new_message[index]])
        else:
            sent_message(["fake"],[new_message[index]])
        index+=1
    while(index< len(new_message)):
        sent_message([],[new_message[index]])
        index+=1


def get_riddle(team_id, riddle_id):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that: 
        1. Once you requested a riddle you cannot request it again per game. 
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle. 
    '''
    payload_sent = {
        'teamId': team_id,
        "riddleId": riddle_id
    }
    response = requests.post(api_base_url+"/fox/get-riddle", json=payload_sent)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        test_case = data['test_case']
    else:
        print("error: ", response.status_code)
    return test_case

def solve_riddle(team_id, solution,total_budget):
    '''
    In this function you will solve the riddle that you have requested. 
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    '''
    payload_sent = {
        'teamId': team_id,
        "solution": solution
    }
    response = requests.post(api_base_url+"/fox/solve-riddle", json=payload_sent)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        budget_increase = data['budget_increase']
        total_budget = data['total_budget']
        status = data['status']
        if(status == "success"):
            print("Riddle solved successfully")
            print("Budget increased by: ", budget_increase)
            print("Total budget: ", total_budget)
    else:
        print("error: ", response.status_code)

def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    '''
    Use this function to call the api end point to send one chunk of the message. 
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call. 
    '''
    payload_sent = {
        'teamId': team_id,
        "messages": messages,
        "message_entities":message_entities
    }
    response = requests.post(api_base_url+"/fox/send-message", json=payload_sent)
    if response.status_code == 200 or response.status_code == 201:
       print("Message sent successfully")
    else:
        print("error: ", response.status_code)
   
def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    '''
    payload_sent = {
        'teamId': team_id,
    }
    response = requests.post(api_base_url+"/fox/end-game", json=payload_sent)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        print("Game ended successfully")
    else:
        print("error: ", response.status_code)
    pass

def submit_fox_attempt(team_id,total_budget):
    '''
     Call this function to start playing as a fox. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve riddles 
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages 
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling 
    '''
    msg, carrier_image=init_fox(team_id)
    ## solve riddles
    for riddle_id,riddle_func in riddle_solvers.items():
        test_case = get_riddle(team_id, riddle_id)
        solution = riddle_func(test_case)
        solve_riddle(team_id, solution,total_budget)
    total_budget=min(total_budget,12)
    ## generate message array
    generate_message_array(msg, carrier_image)
    end_fox(team_id)


submit_fox_attempt(team_id,total_budget)