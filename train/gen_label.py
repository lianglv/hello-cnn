import os

dirs = '/Users/lvliang/ml/hello-cnn/data'
filenames = []
for i in range(3, 10) : 
    dir = os.path.join(dirs, 'exp_%02d' % i) 
    names = os.listdir(dir)
    names = [os.path.join(dir, name) for name in names]
    filenames = filenames + names
filenames = filter(lambda name: 'res' in name, filenames)

with open('train.txt', 'w') as f_train_txt:
    for filename in filenames[:1400]:
        fx, fy = filename[filename.index('_h_') + 3: filename.index('_h_') + 6], filename[filename.index('_w_') + 3: filename.index('_w_') + 6]
        line = '{} {} {}\n'.format(filename, fx, fy)
        f_train_txt.write(line)

with open('val.txt', 'w') as f_val_txt:
    for filename in filenames[1400:2000]:
        fx, fy = filename[filename.index('_h_') + 3: filename.index('_h_') + 6], filename[filename.index('_w_') + 3: filename.index('_w_') + 6]
        line = '{} {} {}\n'.format(filename, fx, fy)
        f_val_txt.write(line)

with open('test.txt', 'w') as f_test_txt:
    for filename in filenames[2000:]:
        line = '{}\n'.format(filename)
        f_test_txt.write(line)
