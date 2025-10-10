import os
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

fpath = os.path.join("/opt/data/private/wjy/SQL/splits/eigen_zhou", "{}_files.txt")
train_filenames = readlines(fpath.format("train_gt"))
val_filenames = readlines(fpath.format("test"))

# 检查测试集和训练集的文件名是否有重复，但需要将训练集的第二项阔成10位，前面补0
train_filenames = [f"{f.split()[0]} {int(f.split()[1]):010d}" for f in train_filenames]
val_filenames = [f"{f.split()[0]} {int(f.split()[1]):010d}" for f in val_filenames]

# 检查训练集和测试集的文件名是否有重复
print("Checking for filename overlap between train and test splits")
train_set = set(train_filenames)
val_set = set(val_filenames)
intersection = train_set.intersection(val_set)