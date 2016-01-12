import datetime

names_file = 'data/dow30.csv'
start = datetime.datetime(1990, 1, 1)
#start = datetime.datetime(2013, 1, 1) # for testing
end = datetime.datetime(2015, 12, 7)
save_file = 'data/dow30_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
normalize_std_len = 50

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 200
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 30


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 650
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 30


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.01
    max_grad_norm = 10
    num_layers = 2
    num_steps = 10
    hidden_size = 1500
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 30
