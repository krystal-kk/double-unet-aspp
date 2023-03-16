import numpy as np
import cv2
import random
import pandas as pd


def random_color():
    color = list(np.random.choice(range(256), size=3))
    return color

def display(cfg):
    print('\nConfigurations:')
    for k, v in vars(cfg).items():
        print('{:30} {}'.format(k, v))
    print('\n')

def get_name_list(train_df, test_df):
    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)
    train_list = train_df['train'].tolist()
    test_list = test_df['test'].tolist()
    return train_list, test_list
