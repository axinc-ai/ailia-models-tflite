# ailia MODELS TFLITE launcher

import os
import sys
import glob
import subprocess

# ======================
# Arguemnt Parser Config
# ======================

sys.path.append('./util')
from utils import get_base_parser, update_parser

parser = get_base_parser(
    'ailia MODELS tflite launchar',
    None,
    None,
)
args = update_parser(parser)

# ======================
# Get model list
# ======================

IGNORE_LIST = []

def get_model_list():
    global model_index

    file_list = []
    for current, subfolders, subfiles in os.walk("./"):
        file_list.append(current)

    file_list.sort()

    model_list = []
    category_list = {}
    model_exist = {}
    for current in file_list:
        current = current.replace("\\", "/")
        files = current.split("/")
        if len(files) == 3:
            if (files[1] in IGNORE_LIST) or (files[2] in IGNORE_LIST):
                continue
            if files[2] in model_exist:
                continue
            script = "./"+files[1]+"/"+files[2]+"/"+files[2]+".py"
            if os.path.exists(script):
                if not(files[1] in category_list):
                    category_list[files[1]] = len(category_list)
                category_id = category_list[files[1]]
                model_list.append({
                    "category": files[1],
                    "category_id": category_id,
                    "model": files[2],
                })
                model_exist[files[2]] = True


    model_name_list = []
    for i in range(len(model_list)):
        model_name_list.append(""+model_list[i]["category"]+" : "+model_list[i]["model"])
        if model_list[i]["model"]=="yolox":
            model_index = i

    return model_list, model_name_list, len(category_list)

# ======================
# Execute model
# ======================

def get_options():
    args_dict = vars(args)
    
    options = []
    for key in args_dict:
        if key=="ftype":
            continue
        if args_dict[key] is not None:
            if args_dict[key] is True:
                options.append("--"+key)
            elif args_dict[key] is False:
                continue
            else:
                options.append("--"+key)
                options.append(str(args_dict[key]))

    if args.input == None and args.video == None:
        options.append("-v")
        options.append("0")
    
    return options

def run_model(model):
    options = get_options()
    
    cmd = sys.executable
    cmd = [cmd, model["model"]+".py"] + options

    print(" ".join(cmd))

    dir = "./"+model["category"]+"/"+model["model"]+"/"

    if args.input != None:
        subprocess.check_call(cmd, cwd=dir, shell=False)
    else:
        proc = subprocess.Popen(cmd, cwd=dir)
        try:
            outs, errs = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            pass

    input('Push enter key to stop')

    proc.kill()
    proc=None

# ======================
# CUI
# ======================

model_list, model_name_list, category_list = get_model_list()

def show_model_list():
    cnt = 0
    for model in model_list:
        print(cnt, model["category"] + " : " + model["model"])
        cnt = cnt + 1

show_model_list()
model_no = input('Number of model (press q to exit): ')
while model_no not in ('q', 'ｑ'):
    try:
        no = int(model_no)
    except:
        no = -1
    if no >= 0 and no < len(model_list):
        run_model(model_list[no])
        show_model_list()
    else:
        print("Invalid model number.")
    model_no = input('Number of model (press q to exit): ')
