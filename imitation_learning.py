import cv2
import numpy as np
import csv

cap = cv2.VideoCapture("training_vid/training_vid.mp4")

red_low = np.array([0, 250, 140])
red_high = np.array([180, 255, 255])
black_low = np.array([0, 0, 25])
black_high = np.array([180, 255, 255])

fps = 30
delta_t = 1/30

padding = 25

width = 360
height = 800

n_back = 3

prev_frame_info = {}

game_on = True
wait_period = 5
cur_wait = None

data = []

data_count = 0

save_data = True
fields = ["s_pos_x_t-n", "s_pos_y_t-n", "s_vel_x_t-n", "s_vel_y_t-n", "puck_pos_x_t-n", "puck_pos_y_t-n", "puck_vel_x_t-n", "puck_vel_y_t-n",
          "s_pos_x_t-n+1", "s_pos_y_t-n+1", "s_vel_x_t-n+1", "s_vel_y_t-n+1", "puck_pos_x_t-n+1", "puck_pos_y_t-n+1", "puck_vel_x_t-n+1", "puck_vel_y_t-n+1",
          "s_pos_x_t-n+2", "s_pos_y_t-n+2", "s_vel_x_t-n+2", "s_vel_y_t-n+2", "puck_pos_x_t-n+2", "puck_pos_y_t-n+2", "puck_vel_x_t-n+2", "puck_vel_y_t-n+2",
          "s_x", "s_y"]

if save_data:
    with open("data.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

H = 288
height = 240
width = 480

cur_state_info = []

try:
    while True:
        ret, img = cap.read()
        img = img[15:H-15, 0: width]

        img = cv2.resize(img, (width, height))

        frame_blur = cv2.GaussianBlur(img, (3,3),0)   

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, red_low, red_high)
        mask_puck = cv2.bitwise_not(cv2.inRange(hsv, black_low, black_high))
        mask_red = cv2.rectangle(mask_red, (0, 0), (height + 10, height), (0, 0, 0), -1)

        cur_frame_info = {}

        for i, mask in enumerate([mask_red, mask_puck]):
            circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=5, param2=10, minRadius=5, maxRadius=50)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x = circle[0]
                    y = circle[1]
                    cur_frame_info[i] = [(round((width-x)/width, 3), round((height-y)/height, 3))]
                    if (None not in prev_frame_info.values()) and (len(prev_frame_info.values()) != 0):
                        sx = round(cur_frame_info[i][0][0] - prev_frame_info[i][0][0], 3)
                        sy = round(cur_frame_info[i][0][1] - prev_frame_info[i][0][1], 3)
                        cur_frame_info[i].append((sx, sy))

                    else:
                        cur_frame_info[i].append((0, 0))

            else:
                cur_frame_info[i] = None

        cur_state_info.append(cur_frame_info)


        if (None in cur_frame_info.values()):
            cur_state_info = [] 
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
                if len(cur_state_info) == n_back+1:
                    cur_data = list(list(cur_state_info[0][0][0])+list(cur_state_info[0][0][1])+
                                    list(cur_state_info[0][1][0])+list(cur_state_info[0][1][1])+
                                    list(cur_state_info[1][0][0])+list(cur_state_info[1][0][1])+
                                    list(cur_state_info[1][1][0])+list(cur_state_info[1][1][1])+
                                    list(cur_state_info[2][0][0])+list(cur_state_info[2][0][1])+
                                    list(cur_state_info[2][1][0])+list(cur_state_info[2][1][1])+
                                    list(cur_state_info[3][0][1]))
                    
                    # print(cur_data)
                    data.append(cur_data)

                    cur_state_info.pop(0)

                    data_count+=1
                    # if data_count % 100 == 0:
                    #     print(data_count)

                    if data_count % 1000 == 0:
                        if save_data:
                            with open("data.csv", 'a', newline='') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerows(data)

                            print(f"Written {data_count} entries....")
                            data = []

            except:
                pass

        prev_frame_info = cur_frame_info

        # cv2.imshow("Frame", img)
        # cv2.waitKey(1)
            
except:
    if save_data:
        with open("data.csv", 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        print(f"Written {data_count} entries....")
        data = []