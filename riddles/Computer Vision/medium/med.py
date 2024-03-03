import numpy as np 
import cv2 
from sklearn.cluster import DBSCAN

	
# Read the query image as query_img 
# and train image This query image 
# is what you need to find in train image 
# Save it in the same directory 
# with the name image.jpg 
query_img = cv2.imread('patch_image.png') 
train_img = cv2.imread('combined_large_image.png') 

# Convert it to grayscale 
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 

# Initialize the ORB detector algorithm 
orb = cv2.ORB_create() 

# Now detect the keypoints and compute 
# the descriptors for the query image 
# and train image 
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

# Initialize the Matcher for matching 
# the keypoints and then match the 
# keypoints 
matcher = cv2.BFMatcher() 
matches = matcher.match(queryDescriptors,trainDescriptors)
good_matches = [m for m in matches if m.distance < 30]

# draw the matches to the final image 
# containing both the images the drawMatches() 
# function takes both images and keypoints 
# and outputs the matched query image with 
# its train image 
matches_img = cv2.drawMatches(query_img, queryKeypoints, 
train_img, trainKeypoints, good_matches[:20],None) 

matches_image_gray = cv2.cvtColor(matches_img, cv2.COLOR_BGR2GRAY)

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

    # Calculate width and height ranges
    width_range = (x, x + w)
    height_range = (y, y + h)
    count=0
    for keypoint in trainKeypoints:
        x, y = keypoint.pt
        if x >= width_range[0] and x <= width_range[1] and y >= height_range[0] and y <= height_range[1]:
            count += 1
    # Append the count to the list
    counts.append(count)
    
counts = np.array(counts)

# Find the contour that has the maximum count of keypoints
max_index = np.argmax(counts)
max_contour = rectangular_contours[max_index]

# Draw the contour on the matches image
matches_image_copy = train_img.copy()
cv2.drawContours(matches_image_copy, rectangular_contours[87:88], -1, (0, 255, 0), 3)
print(cv2.boundingRect(rectangular_contours[87]))

ext_left = np.min(max_contour[:, :, 0])
ext_right = np.max(max_contour[:, :, 0])
ext_top = np.min(max_contour[:, :, 1])
ext_bottom = np.max(max_contour[:, :, 1])

# Create mask for inpainting (optional, set background if needed)
mask = np.zeros(train_img.shape[:2],dtype=np.uint8)  

for y in range(ext_top, ext_bottom + 1):  # Include ext_bottom for inclusive range
    for x in range(ext_left, ext_right + 1):  # Include ext_right for inclusive range
        # Check if pixel is within image boundaries (avoid potential out-of-bounds errors)
        if 0 <= x < train_img.shape[1] and 0 <= y < train_img.shape[0]:  # Skip pixels outside image
            #train_img[y, x] = [1, 1, 1]  # Assuming BGR format, set to white
            mask[y, x] = 255  # Mark the pixel for inpainting (optional)

# Apply inpainting (optional)
if mask.any():  # Check if any pixels were modified
    inpainted_img = cv2.inpaint(train_img, mask, 3, cv2.INPAINT_NS)

else:
    inpainted_img = train_img.copy()

large_image_copy = cv2.drawKeypoints(train_img, trainKeypoints, None, color=(0, 0, 255))

# Display the matches image with the contour and the keypoints
cv2.imshow("Matches Image with Contour and Keypoints", inpainted_img)
cv2.imshow("Matches Image with",matches_image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()