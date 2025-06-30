class Config:
    batch_size = 512
    use_gpu = True
    dataset_mean = (0.1307,)
    dataset_std = (0.3081,)
    epochs = 6
    plot_iter = 10
    lr = 0.01
    momentum = 0.9
    theta1 = 0.5
    theta2 = 0.5
    data_root = './data'