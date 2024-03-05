# Add the necessary imports here
import pandas as pd
import torch
from collections import Counter
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from utils import *
from statsmodels.tsa.arima.model import ARIMA
from array_ml import array_ml_medium
from scipy.spatial.distance import cdist
from collections import Counter
from reedsolo import RSCodec
import zlib
from decoders import DenseDecoder
from critics import BasicCritic
import torch
from torch.optim import Adam
from PIL import Image
import torchvision.transforms as transforms

import warnings 
def solve_cv_easy(test_case: tuple) -> list:
    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image,dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    gray_image = cv2.cvtColor(shredded_image, cv2.COLOR_BGR2GRAY)
    shreds = np.split(gray_image, gray_image.shape[1] // shred_width, axis=1)
    visisted_shreds=set()
    visisted_shreds.add(0)
    def similarity_score(shred1, shred2):
        edge1 = shred1[:, -1]
        edge2 = shred2[:, 0]
        eq = np.equal(edge1, edge2)
        cnt = np.count_nonzero(eq)
        return cnt
    def find_best_match(shred, shreds):
        best_score = 0
        best_index = -1
        for i, s in enumerate(shreds):
            if s is shred:
                continue
            score = similarity_score(shred, s)
            if score > best_score:
                if i in visisted_shreds:
                    continue
                best_score = score
                best_index = i
        visisted_shreds.add(best_index)
        return best_index
    ordered_shreds = []
    shreds_indices=[0]
    current_shred = shreds[0]
    while len(shreds_indices) < len(shreds):
        ordered_shreds.append(current_shred)
        best_index = find_best_match(current_shred, shreds)
        shreds_indices.append(best_index)
        current_shred = shreds[best_index]
    return shreds_indices

def solve_cv_medium(input: tuple) -> list:
    combined_image_array , patch_image_array = input
    combined_image = np.array(combined_image_array,dtype=np.uint8)
    patch_image = np.array(patch_image_array,dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    query_img = patch_image
    train_img = combined_image
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 
    matcher = cv2.BFMatcher() 
    edges = cv2.Canny(train_img_bw, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    counts = []
    def is_rectangle_like(contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return 0.9 <= aspect_ratio <= 1.1
    rectangular_contours = [c for c in contours if is_rectangle_like(c)]
    for contour in rectangular_contours:
        x, y, w, h = cv2.boundingRect(contour)
        width_range = (x, x + w)
        height_range = (y, y + h)
        count=0
        for keypoint in trainKeypoints:
            x, y = keypoint.pt
            if x >= width_range[0] and x <= width_range[1] and y >= height_range[0] and y <= height_range[1]:
                count += 1
        counts.append(count)
    counts = np.array(counts)
    max_index = np.argmax(counts)
    max_contour = rectangular_contours[max_index]
    matches_image_copy = train_img.copy()
    cv2.drawContours(matches_image_copy, rectangular_contours[87:88], -1, (0, 255, 0), 3)
    ext_left = np.min(max_contour[:, :, 0])
    ext_right = np.max(max_contour[:, :, 0])
    ext_top = np.min(max_contour[:, :, 1])
    ext_bottom = np.max(max_contour[:, :, 1])
    mask = np.zeros(train_img.shape[:2],dtype=np.uint8)  

    for y in range(ext_top, ext_bottom + 1):
        for x in range(ext_left, ext_right + 1):
            if 0 <= x < train_img.shape[1] and 0 <= y < train_img.shape[0]:
                mask[y, x] = 255
    if mask.any():
        inpainted_img = cv2.inpaint(train_img, mask, 3, cv2.INPAINT_NS)
    else:
        inpainted_img = train_img.copy()

    return inpainted_img.tolist()

def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = input
    image = np.array(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    return 0


def solve_ml_easy(input) -> list:
    data = pd.DataFrame(input)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    model = ARIMA(data["visits"],order=(7,0,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=50)
    return forecast.tolist()

def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    x=input[0]
    y=input[1]
    point = np.array([[x, y]])
    distances = cdist(point, array_ml_medium, 'euclidean')
    
    min_dist = np.min(distances)
    if(min_dist <=1):
        return 0
    return -1


def solve_sec_medium(input) -> str:
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    tensor_image = transform(input)
    img = tensor_image.unsqueeze(0)
    # img = torch.tensor(input)
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    out = decode(img)

    return out

# print(solve_sec_medium(" "))
def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """

    def hex2bin(s):
        mp = {'0': "0000",
            '1': "0001",
            '2': "0010",
            '3': "0011",
            '4': "0100",
            '5': "0101",
            '6': "0110",
            '7': "0111",
            '8': "1000",
            '9': "1001",
            'A': "1010",
            'B': "1011",
            'C': "1100",
            'D': "1101",
            'E': "1110",
            'F': "1111"}
        bin = ""
        for i in range(len(s)):
            bin = bin + mp[s[i]]
        return bin

    def bin2hex(s):
        mp = {"0000": '0',
            "0001": '1',
            "0010": '2',
            "0011": '3',
            "0100": '4',
            "0101": '5',
            "0110": '6',
            "0111": '7',
            "1000": '8',
            "1001": '9',
            "1010": 'A',
            "1011": 'B',
            "1100": 'C',
            "1101": 'D',
            "1110": 'E',
            "1111": 'F'}
        hex = ""
        for i in range(0, len(s), 4):
            ch = ""
            ch = ch + s[i]
            ch = ch + s[i + 1]
            ch = ch + s[i + 2]
            ch = ch + s[i + 3]
            hex = hex + mp[ch]

        return hex

    def bin2dec(binary):

        binary1 = binary
        decimal, i, n = 0, 0, 0
        while(binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary//10
            i += 1
        return decimal

    def dec2bin(num):
        res = bin(num).replace("0b", "")
        if(len(res) % 4 != 0):
            div = len(res) / 4
            div = int(div)
            counter = (4 * (div + 1)) - len(res)
            for i in range(0, counter):
                res = '0' + res
        return res

    def permute(k, arr, n):
        permutation = ""
        for i in range(0, n):
            permutation = permutation + k[arr[i] - 1]
        return permutation

    def shift_left(k, nth_shifts):
        s = ""
        for i in range(nth_shifts):
            for j in range(1, len(k)):
                s = s + k[j]
            s = s + k[0]
            k = s
            s = ""
        return k

    def xor(a, b):
        ans = ""
        for i in range(len(a)):
            if a[i] == b[i]:
                ans = ans + "0"
            else:
                ans = ans + "1"
        return ans


    initial_perm = [58, 50, 42, 34, 26, 18, 10, 2,
                    60, 52, 44, 36, 28, 20, 12, 4,
                    62, 54, 46, 38, 30, 22, 14, 6,
                    64, 56, 48, 40, 32, 24, 16, 8,
                    57, 49, 41, 33, 25, 17, 9, 1,
                    59, 51, 43, 35, 27, 19, 11, 3,
                    61, 53, 45, 37, 29, 21, 13, 5,
                    63, 55, 47, 39, 31, 23, 15, 7]

    exp_d = [32, 1, 2, 3, 4, 5, 4, 5,
            6, 7, 8, 9, 8, 9, 10, 11,
            12, 13, 12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21, 20, 21,
            22, 23, 24, 25, 24, 25, 26, 27,
            28, 29, 28, 29, 30, 31, 32, 1]

    per = [16, 7, 20, 21,
        29, 12, 28, 17,
        1, 15, 23, 26,
        5, 18, 31, 10,
        2, 8, 24, 14,
        32, 27, 3, 9,
        19, 13, 30, 6,
        22, 11, 4, 25]

    sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

            [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

            [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

            [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

            [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

            [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

            [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

            [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

    final_perm = [40, 8, 48, 16, 56, 24, 64, 32,
                39, 7, 47, 15, 55, 23, 63, 31,
                38, 6, 46, 14, 54, 22, 62, 30,
                37, 5, 45, 13, 53, 21, 61, 29,
                36, 4, 44, 12, 52, 20, 60, 28,
                35, 3, 43, 11, 51, 19, 59, 27,
                34, 2, 42, 10, 50, 18, 58, 26,
                33, 1, 41, 9, 49, 17, 57, 25]


    def encrypt(pt, rkb, rk):
        pt = hex2bin(pt)
        pt = permute(pt, initial_perm, 64)
        left = pt[0:32]
        right = pt[32:64]
        for i in range(0, 16):
            right_expanded = permute(right, exp_d, 48)
            xor_x = xor(right_expanded, rkb[i])
            sbox_str = ""
            for j in range(0, 8):
                row = bin2dec(int(xor_x[j * 6] + xor_x[j * 6 + 5]))
                col = bin2dec(
                    int(xor_x[j * 6 + 1] + xor_x[j * 6 + 2] + xor_x[j * 6 + 3] + xor_x[j * 6 + 4]))
                val = sbox[j][row][col]
                sbox_str = sbox_str + dec2bin(val)

            sbox_str = permute(sbox_str, per, 32)
            result = xor(left, sbox_str)
            left = result
            if(i != 15):
                left, right = right, left
        combine = left + right

        cipher_text = permute(combine, final_perm, 64)
        return cipher_text
    key,pt=input
    key = hex2bin(key)

    keyp = [57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4]

    key = permute(key, keyp, 56)

    shift_table = [1, 1, 2, 2,
                2, 2, 2, 2,
                1, 2, 2, 2,
                2, 2, 2, 1]

    key_comp = [14, 17, 11, 24, 1, 5,
                3, 28, 15, 6, 21, 10,
                23, 19, 12, 4, 26, 8,
                16, 7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55,
                30, 40, 51, 45, 33, 48,
                44, 49, 39, 56, 34, 53,
                46, 42, 50, 36, 29, 32]

    left = key[0:28]
    right = key[28:56]

    rkb = []
    rk = []
    for i in range(0, 16):
        left = shift_left(left, shift_table[i])
        right = shift_left(right, shift_table[i])
        combine_str = left + right
        round_key = permute(combine_str, key_comp, 48)

        rkb.append(round_key)
        rk.append(bin2hex(round_key))

    cipher_text = bin2hex(encrypt(pt, rkb, rk))
    return cipher_text

def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    words,X=input
    word_counts = Counter(words)
    sorted_words = sorted(word_counts, key=lambda word: (-word_counts[word], word))
    return sorted_words[:X]


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    num = []
    str_list = []
    temp = ""
    s=input
    i = 0
    while i < len(s):
        if '0' <= s[i] <= '9':
            n = 0
            while '0' <= s[i] <= '9':
                n = n * 10 + int(s[i])
                i += 1
            i -= 1
            num.append(n)
        elif s[i] == '[':
            str_list.append(temp)
            temp = ""
        elif s[i] == ']':
            t = str_list.pop()
            n = num.pop()
            temp = t + temp * n
        else:
            temp += s[i]
        i += 1
    return temp

def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    def nCr(n, k):
        if k > n - k:
            k = n - k
        ans = 1
        j = 1
        for j in range(1, k + 1):
            if n % j == 0:
                ans *= n // j
            elif ans % j == 0:
                ans = ans // j * n
            else:
                ans = (ans * n) // j
            n -= 1
        return ans
    # solve is the most optimal solution for the given problem O(n)
    x, y = input
    down, right = x - 1, y - 1
    return nCr(down + right, min(right, down))


# riddle_solvers = {
#     'cv_easy': solve_cv_easy,
#     'cv_medium': solve_cv_medium,
#     'cv_hard': solve_cv_hard,
#     'ml_easy': solve_ml_easy,
#     'ml_medium': solve_ml_medium,
#     'sec_medium_stegano': solve_sec_medium,
#     'sec_hard':solve_sec_hard,
#     'problem_solving_easy': solve_problem_solving_easy,
#     'problem_solving_medium': solve_problem_solving_medium,
#     'problem_solving_hard': solve_problem_solving_hard
# }
riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}
