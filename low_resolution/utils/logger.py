import sys, os

import csv, pandas
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
plt.switch_backend('agg')

import numpy as np
import torchvision.utils as vutils


class CSVLogger():
    def __init__(self, every, fieldnames, filename='log.csv', resume=False):
        # If resume, first check if file already exists
        if not os.path.exists(filename):
            resume = False  # if not, proceed as not resuming from anything

        self.every = every
        self.filename = filename
        self.csv_file = open(filename, 'a' if resume else 'w')
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if not resume:
            self.writer.writeheader()
            self.csv_file.flush()

        if resume:
            df = pandas.read_csv(filename)
            if len(df['global_iteration'].values) == 0:
                self.time = 0
            else:
                self.time = df['global_iteration'].values[-1]
        else:
            self.time = 0

    def is_time(self):
        return self.time % self.every == 0

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def plot_csv(csv_path, fig_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        dict_of_lists = {}
        ks = None
        for i, r in enumerate(reader):
            if i == 0:
                for k in r:
                    dict_of_lists[k] = []
                ks = r
            else:
                for _i, v in enumerate(r):
                    try:
                        v = float(v)
                    except:
                        v = 0
                    dict_of_lists[ks[_i]].append(v)
    fig = plt.figure()
    for k in dict_of_lists:
        if k == 'global_iteration':
            continue
        plt.clf()
        plt.plot(dict_of_lists['global_iteration'], dict_of_lists[k])
        plt.title(k)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(fig_path), f'_{k}.jpeg'), bbox_inches='tight', pad_inches=0, format='jpeg')
    
    plt.close(fig)



# __all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()


def save_grid_with_cmap(tensor, save_path, cmap="seismic", nrow=5, padding=1, handle_imgs=False):
    # RdBu / viridis
    
    # tensor = tensor / (tensor.abs().max())
    max_vals = tensor.view(tensor.size(0), -1).abs().max(dim=1)[0]  # shape: (B,)
    max_vals = max_vals.view(-1, 1, 1, 1) + 1e-8  # reshape for broadcasting
    tensor = tensor / max_vals  # shape-preserving normalization

    if handle_imgs:
        grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, scale_each=False, padding=padding, pad_value=0)
        grid_np = grid.cpu().numpy()
        grid_np = np.transpose(grid_np, (1, 2, 0))
        plt.imsave(save_path, grid_np)
    else:
        grid = vutils.make_grid(tensor, nrow=nrow, normalize=False, scale_each=False, padding=padding, pad_value=-1.0)
        grid_np = grid.cpu().numpy()
        if grid_np.shape[0] == 1:
            grid_np = grid_np[0]  # (H, W)
        else:
            grid_np = np.mean(grid_np, axis=0)  

        plt.figure(figsize=(10, 10))
        
        # 1
        # plt.imshow(grid_np, cmap=cmap, norm=colors.CenteredNorm())
        # 2
        vmax = np.max(np.abs(grid_np))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        plt.imshow(grid_np, cmap=cmap, norm=norm)
        
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

