# Import libraries
import numpy as np
import cv2

# Load the shredded image as a NumPy array
shredded_image = np.load("./shredded.jpg")

# Define the shred width
shred_width = 64

# Convert the image to grayscale
gray_image = cv2.cvtColor(shredded_image, cv2.COLOR_BGR2GRAY)

# Split the image into shreds
shreds = np.split(gray_image, gray_image.shape[1] // shred_width, axis=1)

# Define a function to calculate the similarity score between two shreds
def similarity_score(shred1, shred2):
  # Get the right edge of shred1 and the left edge of shred2
  edge1 = shred1[:, -1]
  edge2 = shred2[:, 0]

  # Calculate the sum of absolute differences between the edges
  sad = np.sum(np.abs(edge1 - edge2))

  # Return the similarity score as the inverse of the sad
  return 1 / sad

# Define a function to find the best match for a given shred
def find_best_match(shred, shreds):
  # Initialize the best score and the best index
  best_score = 0
  best_index = -1

  # Loop through the shreds
  for i, s in enumerate(shreds):
    # Skip the shred itself
    if s is shred:
      continue

    # Calculate the similarity score with the shred
    score = similarity_score(shred, s)

    # Update the best score and the best index if the score is higher
    if score > best_score:
      best_score = score
      best_index = i

  # Return the best index
  return best_index

# Define a list to store the ordered shreds
ordered_shreds = []

# Start with the first shred
current_shred = shreds[0]

# Loop until all shreds are ordered
while len(ordered_shreds) < len(shreds):
  # Append the current shred to the ordered shreds
  ordered_shreds.append(current_shred)

  # Find the best match for the current shred
  best_index = find_best_match(current_shred, shreds)

  # Set the current shred to the best match
  current_shred = shreds[best_index]

# Define a list to store the shreds' indices
shreds_indices = []

# Loop through the ordered shreds
for s in ordered_shreds:
  # Find the index of the shred in the original shreds list
  index = shreds.index(s)

  # Append the index to the shreds' indices
  shreds_indices.append(index)

# Print the shreds' indices
print(shreds_indices)
