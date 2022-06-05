import matplotlib.pyplot as plt


def plot_use_plt(mean_loss_list, train_acc_list, test_acc_list, num_epoch):
    x1 = range(0, num_epoch)
    x2 = range(0, num_epoch)
    x3 = range(0, num_epoch)
    plt.subplot(1, 3, 1)
    plt.plot(x1, mean_loss_list, 'o-')
    plt.title('Train_loss vs.epochs')
    plt.ylabel('Train loss')
    plt.subplot(1, 3, 2)
    plt.plot(x2, train_acc_list, '.-')
    plt.title('Train_acc vs.epochs')
    plt.ylabel('Train acc')
    plt.subplot(1, 3, 3)
    plt.plot(x3, test_acc_list, '.-')
    plt.title('Test_acc vs.epochs')
    plt.ylabel('Test acc')
    plt.savefig("./training_results/show.jpg")#这一句话一定要放在plt.show()前面
    plt.show()