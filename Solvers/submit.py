import requests
import numpy as np
from LSBSteg import encode,LSBSteg
from riddle_solvers import *
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import time

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

session = requests.Session()
api_base_url = "http://16.171.171.147:5000"
# api_base_url = "http://localhost:3005"
# team_id = Lu2xdzj (take care to use the same team id and start game 🧑🏼‍🚒)
team_id="Lu2xdzj"
# team_id = "xxx"
total_budget=0
def init_fox(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    payload_sent = {
        'teamId': team_id
    }
    response = session.post(api_base_url+"/fox/start", json=payload_sent)
    print(response)
    if response.status_code == 200 or response.status_code == 201:
        print("Game started successfully")
        data = response.json()
        msg = data['msg']
        carrier_image = data['carrier_image']
        return msg, np.array(carrier_image),np.array(carrier_image,dtype=np.uint8)
    else:
        print("error: ", response.status_code)
        return None, None,None



def split_string_into_two_chars(input_string):
    pairs = [input_string[i:i+2] for i in range(0, len(input_string), 2)]
    if len(input_string) % 2 != 0:
        pairs.append(pairs.pop()[0])
    return pairs
    
def send_message(team_id, messages, message_entities=['F', 'F', 'R']):
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
    response = session.post(api_base_url+"/fox/send-message", json=payload_sent)
    # if response.status_code == 200 or response.status_code == 201:
    #    print("Message sent successfully")
    # else:
    #     print("error: ", response.status_code)
   
def prepare_message(fake_msg,real_msg,total_budget,channel,team_id,image_carrier):
    message_entities = ['E' for _ in range(3)]
    messages = [image_carrier.tolist() for _ in range(3)]
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
    send_message(team_id, messages, message_entities)
    return channel,total_budget

def generate_message_array(message, image_carrier, total_budget, team_id):
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''
    # new_message = split_string_into_two_chars(message)
    new_message =[message]
    index=0
    channel=0

    # if(total_budget>=2):
        # print("total_budget > 2")
    channel,total_budget = prepare_message(["$#$#$","#$#$$$"],[new_message[index]],total_budget,channel,team_id,image_carrier)
        
    # elif (total_budget>=1):
    #     print("total_budget > 1")
    #     channel,total_budget = prepare_message(["#$#$"],[new_message[index]],total_budget,channel,team_id,image_carrier)
    # else:
    #     print("total_budget == 0")
    #     channel,total_budget = prepare_message([],[new_message[index]],total_budget,channel,team_id,image_carrier)

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
    response = session.post(api_base_url+"/fox/get-riddle", json=payload_sent)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        # print("Riddle requested successfully")
        test_case = data['test_case']
        return test_case
    else:
        print("error: ", response.status_code)
        return ''

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
    response = session.post(api_base_url+"/fox/solve-riddle", json=payload_sent)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        budget_increase = data['budget_increase']
        total_budget = data['total_budget']
        status = data['status']
        # if(status == "success"):
        #     # print("Riddle solved successfully")
        #     # print("Budget increased by: ", budget_increase)
        #     # print("Total budget: ", total_budget)
        # else:
            # print("Riddle not solved")
    else:
        print("error: ", response.status_code)
    return total_budget

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
    response = session.post(api_base_url+"/fox/end-game", json=payload_sent)
    
    if response.status_code == 200 or response.status_code == 201:
        print("Game ended successfully")
    else:
        print("error: ", response.status_code)
    return response


def fail_riddle(team_id):
    payload_sent = {
        'teamId': team_id,
        "solution": "faile"
    }
    response = session.post(api_base_url+"/fox/solve-riddle", json=payload_sent)
    print(response)
    


msg, carrier_image,carrier_image2=init_fox(team_id)

start_time = time.time()
diffrences=[]

riddle_id="problem_solving_hard"
test_case_problem_solving_hard = get_riddle(team_id, riddle_id)
# print(test_case_problem_solving_hard)
solution_1 = solve_problem_solving_hard(test_case_problem_solving_hard)
total_budget = solve_riddle(team_id, solution_1,total_budget)
end_time1= time.time()
diffrences.append(end_time1-start_time)



riddle_id="problem_solving_easy"
test_case_problem_solving_easy = get_riddle(team_id, riddle_id)
# print(test_case_problem_solving_easy)
solution_2 = solve_problem_solving_easy(test_case_problem_solving_easy)
total_budget = solve_riddle(team_id, solution_2,total_budget)
end_time2= time.time()
diffrences.append(end_time2-end_time1)




riddle_id="problem_solving_medium"
test_case_problem_solving_medium = get_riddle(team_id, riddle_id)
# print(test_case_problem_solving_medium)
solution_3 = solve_problem_solving_medium(test_case_problem_solving_medium)
total_budget = solve_riddle(team_id, solution_3,total_budget)
end_time3= time.time()
diffrences.append(end_time3-end_time2)



riddle_id="sec_hard"
test_case_sec_hard = get_riddle(team_id, riddle_id)
# print(test_case_sec_hard)
solution_4 = solve_sec_hard(test_case_sec_hard)
total_budget = solve_riddle(team_id, solution_4,total_budget)
end_time4= time.time()
diffrences.append(end_time4-end_time3)




riddle_id="cv_easy"
test_case_cv_easy = get_riddle(team_id, riddle_id)
# print(test_case_cv_easy)
solution_5 = solve_cv_easy(test_case_cv_easy)
total_budget = solve_riddle(team_id, solution_5,total_budget)
end_time5= time.time()
diffrences.append(end_time5-end_time4)



riddle_id="ml_easy"
test_case_ml_easy = get_riddle(team_id, riddle_id)
# print(test_case_ml_easy)
solution_6 = solve_ml_easy(test_case_ml_easy)
total_budget = solve_riddle(team_id, solution_6,total_budget)
end_time6= time.time()
diffrences.append(end_time6-end_time5)





riddle_id="ml_medium"
test_case_ml_medium = get_riddle(team_id, riddle_id)
# print(test_case_ml_medium)
solution_7 = solve_ml_medium(test_case_ml_medium)
total_budget = solve_riddle(team_id, solution_7,total_budget)
end_time7= time.time()
diffrences.append(end_time7-end_time6)





# try:
riddle_id="sec_medium_stegano"
test_case_sec_medium_stegano = get_riddle(team_id, riddle_id)
# print(test_case_sec_medium_stegano)
solution_9 =solve_sec_medium( np.transpose(test_case_sec_medium_stegano[0], (1, 2, 0)) ) 
total_budget = solve_riddle(team_id, solution_9,total_budget)
end_time8 = time.time()
diffrences.append(end_time8-end_time7)



riddle_id="cv_hard"
test_case_cv_hard = get_riddle(team_id, riddle_id)
# print(test_case_sec_medium_stegano)
solution_10 =solve_cv_hard( test_case_cv_hard,processor,model ) 
total_budget = solve_riddle(team_id, solution_10,total_budget)
end_time9= time.time()
diffrences.append(end_time9-end_time8)

# except Exception as e:
#     payload_sent = {
#             'teamId': team_id,
#             "solution":2
#         }
#     response = session.post(api_base_url+"/fox/solve-riddle", json=payload_sent)
#     print('error in test_case_cv_hard')
#     print(e)
    
# try:
#     riddle_id="cv_medium"
#     test_case_cv_medium = get_riddle(team_id, riddle_id)
#     solution_11 =solve_cv_medium( test_case_cv_medium ) 
#     total_budget = solve_riddle(team_id, solution_11,total_budget)
# except Exception as e:
#     payload_sent = {
#             'teamId': team_id,
#             "solution": []
#     }
#     response = requests.post(api_base_url+"/fox/solve-riddle", json=payload_sent)
#     print('error in cv_medium')
#     print(e)

# steg = LSBSteg(carrier_image)

# 1
new_message =[msg[:3]]
index=0
channel=0
channel,total_budget = prepare_message([""],[new_message[index]],total_budget,channel,team_id,carrier_image)


# 2
new_message =[msg[3:6]]
index=0
channel=0
channel,total_budget = prepare_message([""],[new_message[index]],total_budget,channel,team_id,carrier_image)



# 3
new_message =[msg[6:9]]
index=0
channel=0
channel,total_budget = prepare_message([""],[new_message[index]],total_budget,channel,team_id,carrier_image)



# 4
new_message =[msg[9:12]]
index=0
channel=0
channel,total_budget = prepare_message([""],[new_message[index]],total_budget,channel,team_id,carrier_image)



# 5
new_message =[msg[12:15]]
index=0
channel=0
channel,total_budget = prepare_message([""],[new_message[index]],total_budget,channel,team_id,carrier_image)




# 6
new_message =[msg[15:]]
index=0
channel=0
channel,total_budget = prepare_message([""],[new_message[index]],total_budget,channel,team_id,carrier_image)



rsponse_end = end_fox("Lu2xdzj")
end_time = time.time()
print(rsponse_end.text)
elapsed_time = end_time - start_time

# Print the result
print("Elapsed time: {:.4f} seconds".format(elapsed_time))
cv2.imwrite('fox_trial3/carrier_image.png',carrier_image)
cv2.imwrite('fox_trial3/cv_hard.png',np.array(test_case_cv_hard[1]))
print("cv hard test",test_case_cv_hard[0])
print("the solution of cv hard is ",solution_10)
print("the message is : ",msg)
print('problem solving hard test',test_case_problem_solving_hard)
print(diffrences)
# save imagenp.transpose(test_case_sec_medium_stegano[0], (1, 2, 0))
