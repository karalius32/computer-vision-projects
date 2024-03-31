This is my first computer vision project I made with a goal of learning some basics in this field. I use YOLO8 model to detect cars, trucks, buses and motorcycles and count them when they cross the red line.
![main](https://github.com/karalius32/computer-vision-projects/assets/59309454/923134cb-2ee4-4419-adbb-5e8d6efea826)
I also use custom made tracker for object tracking. It works by moving bounding box of last frame by some pixels in the direction of the road, then calculating it's iou score with each bounding box of current score and comparing it with the treshold. Problem with this aproach is that there are two values that needs to be hard coded if there would be a need to use this project with another video. This could be solved by using ML model to predict the direction cars are moving by.
<br>
<br>
Other problems that need to be solved:<br>
-YOLO8 should be additionally trained with road data to be able detect motorcycles better and to distinguish between cars and trucks.<br>
-Object tracker fails if bounding box is small in at least one dimension.
This example illustrates these two problems:<br>
![Untitled design](https://github.com/karalius32/computer-vision-projects/assets/59309454/fb339207-17eb-470a-a672-a96bfad87e9f)



