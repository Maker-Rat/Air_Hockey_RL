import cv2
import numpy as np
import csv

cap = cv2.VideoCapture("training_videos/training_vid_1.mp4")

yellow_low = np.array([20, 100, 100])
yellow_high = np.array([75, 255, 255])

combined_low = np.array([0, 120, 0])
combined_high = np.array([255, 255, 255])

blue_low = np.array([80, 120, 0])
blue_high = np.array([135, 255, 255])

fps = 30
delta_t = 1/30

padding = 25

width = 360
height = 800

prev_state_info = {}

game_on = True
wait_period = 5
cur_wait = None

data = []

data_count = 0

save_data = True

fields = ["s1_pos_x", "s1_pos_y", "s1_vel_x", "s1_vel_y","s2_pos_x", "s2_pos_y", "s2_vel_x", "s2_vel_y",
          "puck_pos_x", "puck_pos_y", "puck_vel_x", "puck_vel_y", "s1_ax", "s1_ay"]

if save_data:
    with open("data_full.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

try:
    while True:
        ret, img = cap.read()

        frame = cv2.resize(img, (width, height))

        frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

        frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)

        img_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(img_hsv, yellow_low, yellow_high)
        mask_combined = cv2.inRange(img_hsv, combined_low, combined_high)
        mask_blue = cv2.inRange(img_hsv, blue_low, blue_high)

        mask_red = cv2.bitwise_xor(mask_combined, cv2.bitwise_or(mask_blue, mask_yellow))

        cur_state_info = {}

        for i, mask in enumerate([mask_blue, mask_red, mask_yellow]):
            res_img = cv2.bitwise_and(frame_blur,frame_blur, mask=mask)
            img_s_gray = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
            canny_edge = cv2.Canny(img_s_gray, 50, 240)

            circles = cv2.HoughCircles(canny_edge, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=5, param2=10, minRadius=10, maxRadius=50)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x = circle[0]
                    y = circle[1]
                    cur_state_info[i] = [(round((x-padding)/width, 3), round((y-padding)/height, 3))]
                    if (None not in prev_state_info.values()) and (len(prev_state_info.values()) != 0):
                        vx = round((cur_state_info[i][0][0] - prev_state_info[i][0][0])/delta_t, 3)
                        vy = round((cur_state_info[i][0][1] - prev_state_info[i][0][1])/delta_t, 3)
                        cur_state_info[i].append((vx, vy))

                    else:
                        cur_state_info[i].append((0, 0))

            else:
                cur_state_info[i] = None

        if (None in cur_state_info.values()):
            if game_on:
                game_on = False

        else:
            if not game_on:
                if cur_wait is None:
                    cur_wait = wait_period
                else:
                    if not cur_wait:
                        cur_wait -= 1
                    else:
                        cur_wait = None
                        game_on = True


        if game_on:
            try:
                cur_data = list(list(prev_state_info[0][0])+list(prev_state_info[0][1])+
                           list(prev_state_info[1][0])+list(prev_state_info[1][1])+
                           list(prev_state_info[2][0])+list(prev_state_info[2][1])+list(cur_state_info[0][1]))
                
                data.append(cur_data)

                data_count+=1
                print(data_count)

                if data_count % 500 == 0:
                    if save_data:
                        with open("data_full.csv", 'a', newline='') as csvfile:
                            csvwriter = csv.writer(csvfile)
                            csvwriter.writerows(data)

                        print(f"Written {data_count} entries....")
                        data = []

            except:
                pass

        prev_state_info = cur_state_info

        #cv2.imshow("Frame", frame)
        #cv2.waitKey(1)
            
except:
    if save_data:
        with open("data_full.csv", 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        print(f"Written {data_count} entries....")
        data = []