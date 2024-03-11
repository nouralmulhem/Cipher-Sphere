<div align= >

# Cipher Sphere

</div>
<div align="center">
   <img align="center" height="350px"  src="https://i.pinimg.com/originals/7b/99/67/7b9967bab38c9140f472c16b6d7c1d0c.gif" alt="logo">
   <br>

### ”🦊 Fox VS Eagle 🦅“

</div>

<p align="center"> 
    <br> 
</p>

## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents

- <a href ="#about"> 📙 Overview</a>
- <a href ="#Started"> 💻 Get Started</a>
- <a href ="#Modules">🤖 Modules</a>
  - <a href="#fox">🔁 Fox Module</a>
  - <a href="#eagle">💪 Eagle Module</a>
- <a href ="#Contributors"> ✨ Contributors</a>
- <a href ="#License"> 🔒 License</a>
<hr style="background-color: #4b4c60"></hr>

<a id = "about"></a>

## <img align="center"  height =50px src="https://user-images.githubusercontent.com/71986226/154076110-1233d7a8-92c2-4d79-82c1-30e278aa518a.gif"> Overview

<ul>
<li> Our solution for <a href="https://www.dell.com/en-eg/dt/microsites/hacktrick.htm?dgc=SM&cid=1083545&lid=spr12198213058&refid=sm_ADVOCACY_spr12198213058&linkId=258142432#collapse&tab0=0&%23eligibility">Dell Hacktrick 2024</a></li>
<li> This project ranked <strong>5th place</strong> among 35 teams in leaderboard.</li>

<li> Built using <a href="https://docs.python.org/3/">Python</a>.</li>
<li>Game Description
<ul>
<li>Firstly: The Fox. Mischievous, and sly, the Fox uses all its tactics to fool the Eagle and
try to send the message through to the parrot using steganography. As the Fox, you’ll have
the opportunity to invest your time wisely into honing your skills and creating distractions
to increase your chances of evading the Eagle’s watchful gaze.</li>
<li>Second: The Eagle. Sharp-eyed and vigilant, the Eagle uses its attentiveness to try to
intercept and decode the messages sent without getting fooled. Beware of the Fox’s devious
tricks, for Fake messages may cross your path. Your mission is to distinguish truth from deception, ensuring that only genuine messages are intercepted while avoiding costly mistakes.
The parrot represents the game administrator that receives the messages and scores both
ends accordingl</li>
<br>

</ul>
</li>
</ul>
<hr style="background-color: #4b4c60"></hr>
<a id = "Started"></a>

## <img  align= center width=50px height=50px src="https://c.tenor.com/HgX89Yku5V4AAAAi/to-the-moon.gif"> How To Run

- First install the <a href="https://github.com/nouralmulhem/Cipher-Sphere/blob/main/requirements.txt">needed packages</a>.</li>

```sh
pip install -r requirements.txt
```

- Folder Structure

```sh
├─── data
├─── Documentation
│   ├── API Documentation.pdf
│   ├── Hackathon General Documentation.pdf
│   └─── Riddles Documentation.pdf
├─── Eagle
│   ├── eagle.py
│   ├── Eagle_submission_script.ipynb
│   ├── BiLstm_code.ipynb
│   └─── GRU_code.ipynb
├─── Solvers
│   ├─── fox_submission_solver.py
│   └─── eagle_submission_solver.py
....
```

<hr style="background-color: #4b4c60"></hr>
<a id = "Modules"></a>

## <img  align= center width=60px src="https://media0.giphy.com/media/j3nq3JkXp0bkFXcNlE/giphy.gif?cid=ecf05e47cftu8uth80woqhyl1kr7oy4m7zaihotdf9twrcaa&ep=v1_stickers_search&rid=giphy.gif&ct=s"> Modules

<a id = "fox"></a>

### <img align= center width=90px src="https://i.giphy.com/IbjRAoXxWCmiY.webp">Fox Module

<ol>
<li>The primary objective for the Fox is to send the secret message to the parrot, encoded through steganography, while devising a strategic game plan to outsmart the Eagle and prevent it from intercepting the message. </li>
<li>For each game, you will be provided with a message
of a specific length and an image to use for encoding the messages. During the first phase,
the message length is fixed to 20 characters.
</li>
<li>
The game is played in chunks (anything between 1 and 20). In every chunk, there are 3
channels that concurrently carry your messages. The messages can be one of the following
3 types:
<ul>
<li>
Real: These messages are part of the original message intended for transmission.
</li>
<li>
Empty: These are Empty messages.
</li>
<li>
Fake: These are Fake messages used to deceive the Eagle. Unlocking this feature requires solving riddles
</li>
</ul>
</li>
<br>
<div align="center">
   <img align="center" height="350px"  src="https://github.com/EslamAsHhraf/Recipe-Frontend/assets/71986226/a7c84601-ff57-455f-8a19-8f8ed7e71b31" alt="logo">

</div>
<br>

<li>Riddles: Whenever you successfully solve a riddle, you will receive a specific reward based on the
difficulty level of the riddle. The rewards for each difficulty are as follows:
<br>
<table>
<tr>
<th>Riddle Type</th>
<th>Budget</th>
</tr>
<tr>
<td>Easy</td>
<td>A budget of 1 Fake message</td>
</tr>
<tr>
<td>Medium</td>
<td>A budget of 2 Fake message</td>
</tr>
<tr>
<td>Hard</td>
<td>A budget of 3 Fake message</td>
</tr>
</table>
There is a total of 10 different riddles:
<ul>
<li>2 Security Riddles (medium and hard).</li>
<li>3 Computer Vision Riddles (easy, medium, and hard difficulties)</li>
<li>2 Machine Learning Riddles (easy and medium difficulties).
</li>
<li>3 Problem Solving Riddles (easy, medium, and hard difficulties).
</li>
</ul>
</li>
<br>
<div align="center">
   <img align="center" src="https://github.com/EslamAsHhraf/Recipe-Frontend/assets/71986226/5f46ed9d-ca91-4d0e-9cf3-3f0f5ec9394f" alt="logo">

</div>
</ol>
<a id = "eagle"></a>

### <img align= center width=50px src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGJoMDJrbnJnaHd1NmU0dDJ0MDB2YmZ6b2tlam53OHpuMTIzcjlldCZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/Rlxa6SJwyW3otUiB8E/giphy.webp">Eagle Module
<ol>
<li>As previously mentioned, there are three channels through which messages are sent
simultaneously at a time. The channels receive messages in the same order sent by the
Fox. However, you can only intercept one channel at a time, which means you will miss
the messages sent on the other two channels.</li>
<li>
To assist in this identification process, you receive three footprints at a time, with each
footprint corresponding to one of the three channels. These footprints indicate whether the
message on a specific channel is Real, Fake, or Empty. By analyzing the footprints, you
can determine which message is genuine and request it from that channel. However, be
cautious, as requesting a message that is Empty or Fake would result in a penalty and could
lead to missing a Real message altogether!</li>
<li>Each footprint is a visual representation of an audio file, one for each channel. Your
task is to identify the word represented, being one of: ”Dell”, Fooled, or just an Empty
audio with some random noise. Rather than working directly with raw audio files, we’ve
transformed them into spectrograms with fixed dimensions (Tx, Nfreq).
</li>
<br>
<div align="center">
   <img align="center" height="350px"  src="https://github.com/EslamAsHhraf/Recipe-Frontend/assets/71986226/cce9f14e-b299-440e-a871-45f5b091b701" alt="logo">

</div>
<br>
<li>Once you have requested and received a message, you need to decode it using the Least
Significant Bit (LSB) method explained earlier and submit the decoded message. The
accuracy of the submitted message will be verified and contribute to your score. It is crucial to remember that after requesting a message, you must submit a message in response.
Failure to do so will cause the game to enter a frozen state until the timeout is reached, and
the game ends.</li>
<br>
<div align="center">
   <img align="center" src="https://github.com/EslamAsHhraf/Recipe-Frontend/assets/71986226/6afd8a80-a7e5-4c72-b456-7095c28d6f22" alt="logo">

</div>
</ol>
<hr style="background-color: #4b4c60"></hr>

<a id ="Contributors"></a>

## <img  align="center" width= 70px height =55px src="https://media0.giphy.com/media/Xy702eMOiGGPzk4Zkd/giphy.gif?cid=ecf05e475vmf48k83bvzye3w2m2xl03iyem3tkuw2krpkb7k&rid=giphy.gif&ct=s"> Contributors

<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/nouralmulhem"><img src="https://avatars.githubusercontent.com/u/76218033?v=4" width="150;" alt=""/><br /><sub><b>Nour Almulhem</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/ahmedmadbouly186" ><img src="https://avatars.githubusercontent.com/u/66012617?v=4" width="150;" alt=""/><br /><sub><b>Ahmed Madbouly</b></sub></a><br />
    </td>
       <td align="center"><a href="https://github.com/Ahmed-H300"><img src="https://avatars.githubusercontent.com/u/67925988?v=4" width="150;" alt=""/><br /><sub><b>Ahmed Hany</b></sub></a><br /></td>
    </td>
       <td align="center"><a href="https://github.com/MohamedNasser8"><img src="https://avatars.githubusercontent.com/u/66921605?v=4" width="150;" alt=""/><br /><sub><b>Mohamed Nasser</b></sub></a><br /></td>
     <td align="center"><a href="https://github.com/EslamAsHhraf"><img src="https://avatars.githubusercontent.com/u/71986226?v=4" width="150;" alt=""/><br /><sub><b>Eslam Ashraf</b></sub></a><br /></td>
  </tr>
</table>

<a id ="License"></a>

## 🔒 License

> **Note**: This software is licensed under MIT License, See [License](https://github.com/nouralmulhem/Cipher-Sphere/blob/main/LICENSE) for more information ©nouralmulhem.
