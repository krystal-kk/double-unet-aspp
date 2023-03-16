import os
import matplotlib.pyplot as plt

def draw_loss(train_loss, test_loss, weight_dir):
    
    # only train loss
    plt.figure(figsize=(8, 6))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss, label='train')
    plt.title('Training loss')
    plt.legend()
    plt.savefig(os.path.join(weight_dir, 'train loss.png'))
    
    # train and test loss
    plt.figure(figsize=(8, 6))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.title('Training and test loss')
    plt.legend()
    plt.savefig(os.path.join(weight_dir, 'train and test loss.png'))
