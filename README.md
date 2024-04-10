# NAO_Gesture_Generation
This repo contains the codes for gesture generation which can be applied to the NAO robot.



Dirctional_Vec_Info:

    dir_vec_pairs = [
        (0, 1, 0.26), # 0, spine-neck
        (1, 2, 0.22), # 1, neck-left shoulder
        (1, 3, 0.22), # 2, neck-right shoulder
        (2, 4, 0.36), # 3, left shoulder-elbow
        (4, 6, 0.33), # 4, left elbow-wrist
        (6, 8, 0.137), # 5 wrist-left index 1
        (8, 9, 0.044), # 6
        (9, 10, 0.031), # 7
    
        (6, 11, 0.144), # 8 wrist-left middle 1
        (11, 12, 0.042), # 9
        (12, 13, 0.033), # 10
    
        (6, 14, 0.127), # 11 wrist-left pinky 1
        (14, 15, 0.027), # 12
        (15, 16, 0.026), # 13
    
        (6, 17, 0.134), # 14 wrist-left ring 1
        (17, 18, 0.039), # 15
        (18, 19, 0.033), # 16
    
        (6, 20, 0.068), # 17 wrist-left thumb 1
        (20, 21, 0.042), # 18
        (21, 22, 0.036), # 19
    
        (3, 5, 0.36), # 20, right shoulder-elbow
        (5, 7, 0.33), # 21, right elbow-wrist
    
        (7, 23, 0.137), # 22 wrist-right index 1
        (23, 24, 0.044), # 23
        (24, 25, 0.031), # 24
    
        (7, 26, 0.144), # 25 wrist-right middle 1
        (26, 27, 0.042), # 26
        (27, 28, 0.033), # 27
    
        (7, 29, 0.127), # 28 wrist-right pinky 1
        (29, 30, 0.027), # 29
        (30, 31, 0.026), # 30
    
        (7, 32, 0.134), # 31 wrist-right ring 1
        (32, 33, 0.039), # 32
        (33, 34, 0.033), # 33
    
        (7, 35, 0.068), # 34 wrist-right thumb 1
        (35, 36, 0.042), # 35
        (36, 37, 0.036), # 36
    
        (1, 38, 0.18), # 37, neck-nose
        (38, 39, 0.14), # 38, nose-right eye
        (38, 40, 0.14), # 39, nose-left eye
        (39, 41, 0.15), # 40, right eye-right ear
        (40, 42, 0.15), # 41, left eye-left ear
    ]
