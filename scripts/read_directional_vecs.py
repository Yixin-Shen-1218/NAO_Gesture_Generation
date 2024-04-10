#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use rest Method"""
import numpy as np
import pandas as pd
import qi
import argparse
import sys


vec_name = ["spine-neck", "neck-left shoulder", "neck-right shoulder", "left shoulder-elbow", "left elbow-wrist",
"wrist-left index 1", "wrist-left index 2", "wrist-left index 3", "wrist-left middle 1", "wrist-left middle 2",
"wrist-left middle 3", "wrist-left pinky 1", "wrist-left pinky 2", "wrist-left pinky 3", "wrist-left ring 1",
"wrist-left ring 2", "wrist-left ring 3", "wrist-left thumb 1", "wrist-left thumb 2", "wrist-left thumb 3",
"right shoulder-elbow", "right elbow-wrist", "wrist-right index 1", "wrist-right index 2", "wrist-right index 3",
"wrist-right middle 1", "wrist-right middle 2", "wrist-right middle 3", "wrist-right pinky 1", "wrist-right pinky 2",
"wrist-right pinky 3", "wrist-right ring 1", "wrist-right ring 2", "wrist-right ring 3", "wrist-right thumb 1",
"wrist-right thumb 2", "wrist-right thumb 3", "neck-nose", "nose-right eye", "nose-left eye", "right eye-right ear",
"left eye-left ear"
]

def clamp_matrix(processed_matrix):
    """
    Rounds a value to -1 or 1 if it's not within the range [-1, 1].
    """
    # Find elements outside the range [-1, 1]
    mask_lower = processed_matrix < -1
    mask_upper = processed_matrix > 1

    # Clamp elements to the range [-1, 1]
    processed_matrix[mask_lower] = -1
    processed_matrix[mask_upper] = 1

    # Round elements to the nearest integer
    # processed_matrix = np.round(processed_matrix)

    return processed_matrix
def main():
    obj = pd.read_pickle("../generation_results/o1rERZRFyqE_112_4_0.pkl")
    directional_vecs = obj['out_dir_vec']
    directional_vecs = directional_vecs.reshape(directional_vecs.shape[0], 42, 3)
    # print directional_vecs[0: 15]

    num = 0

    for directional_vec in directional_vecs:
        if num == 0 or num == 1:
            print ("--------------------num = {}--------------------".format(num))
            for i, vec in enumerate(directional_vec):
                print vec_name[i], vec
        num += 1

    directional_vecs = clamp_matrix(directional_vecs)
    # print directional_vecs
    print directional_vecs.shape

    num = 0

    for directional_vec in directional_vecs:
        if num == 0 or num == 1:
            print ("--------------------num = {}--------------------".format(num))
            for i, vec in enumerate(directional_vec):
                print vec_name[i], vec
        num += 1

    # print len(vec_name)


    # for name in vec_name:
    #     print name

if __name__ == "__main__":
    main()