import sys
from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

### for mac only. For windows, change accordingly
mpl.use('macosx')
######################################################
# * 操作指南
# 运行前修改 文件路径，保存label路径
#  运行后，会展示图片。
#  过程中 按 0, 1, 2, 3, 保存对应label；按了之后，会跳下一张
#  按 esc: 停止并退出，保存当前label过的。下次运行会从下一个没标的开始
# *
######################################################
### must has slash / at the end
my_folder = "val/s9/"
work_saved_in = "temp_result.json"
img_size = (15, 9)
CHECK_MODE = True
######################################################################################################
# read file list
files = [f for f in listdir(my_folder) if isfile(join(my_folder, f)) and "png" in f]
filenames = [f.replace(".png", "") for f in files]
assert len(filenames) == len(files)
# load pre work
if isfile(work_saved_in):
    print("exist unfinished work!!!")
    with open(work_saved_in) as json_file:
        data = json.load(json_file)
else:
    print("start from scratch!!!")
    data = {}


def save_work():
    with open(work_saved_in, 'w') as outfile:
        json.dump(data, outfile)
        print(f"[save in {work_saved_in}]")


# show img
def record_label(event):
    sys.stdout.flush()
    if event.key in ["0", "1", "2", "3"] and not CHECK_MODE:
        global data, label_key_
        print(f'[label] {label_key_} - ', event.key)

        data[label_key_] = int(event.key)
        plt.close()
    elif event.key == 'escape':
        plt.close()
        print("stop labelling!!! Exiting")
        save_work()
        global continue_flag_
        continue_flag_ = False
        # sys.exit(0)
    elif event.key == 'n':
        plt.close()
    else:
        print('only 0,1,2,3 are valid!!!')

continue_flag_ = True
label_dict={0:"default",1:"Hand points up ",2:"hand points down",3:"hand points right ->" }
for png_name in filenames:
    if not continue_flag_:
        print("Exiting. out")
        break
    label_key_ = f"{'_'.join(my_folder.split('/'))}{png_name}"
    if label_key_ in data and not CHECK_MODE:
        continue
    print(f"showing {label_key_}")
    img = mpimg.imread(f'{my_folder}/{png_name}.png')
    fig = plt.figure(figsize=img_size)
    if CHECK_MODE:
        plt.title(f"label:  {label_dict[data[label_key_]]}")

        ## 强迫症 确认label key 没问题, 这里作用域不是那么正规
    label_key_ = f"{'_'.join(my_folder.split('/'))}{png_name}"
    fig.canvas.mpl_connect('key_press_event', record_label)
    imgplot = plt.imshow(img)
    plt.show()

save_work()