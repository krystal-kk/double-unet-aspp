import os
import random
import shutil


"""split the whole dataset into train and test"""
def move_file(file_path, move_path):
    file_list = os.listdir(file_path)
    all_num = len(file_list)
    ratio = 0.3
    random.seed(12)
    pick_num = int(all_num * ratio)
    sample = random.sample(file_list, pick_num)
    print(len(sample))
    # for i in sample:
    #     shutil.move(os.path.join(file_path, i), os.path.join(move_path, i))



if __name__ == '__main__':
    file_path = '/home/krystal/workspace/dataset/cell/all_data/images'
    move_path = '/Users/xuexi/Desktop/12'
    move_file(file_path, move_path)
