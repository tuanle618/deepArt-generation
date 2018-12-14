# -*- coding: utf-8 -*-
"""
@title create_batch_file.py
@author: Tuan Le
@email: tuanle@hotmail.de
"""
import sys
def write_batch_file(model, start_epoch=0, epochs=500, batch_size=16, save_intervals=50, final_epoch=10000):
    #init batch_file
    batch_file = open("train_cycle_model-{}_start-{}_end-{}.bat".format(model, start_epoch, final_epoch), "w")
    #write first line:
    batch_file.write("@echo off")
    batch_file.write("\n")
    steps = (final_epoch-start_epoch) // epochs
    iterator = range(steps)
    python_path = sys.executable
    script_path = r"E:\deepArt-generation\source\train_model.py"
    base_cycle = start_epoch / epochs
    for i in iterator:
        cycle = int(base_cycle + i + 1)
        if i == 0 and start_epoch == 0:
            init_train = True
            cmd_string = "{} {} {} {} {} {} {} {} {}".format(python_path, script_path, model, init_train, start_epoch, cycle, epochs, batch_size, save_intervals)
        else:
            init_train = False
            cmd_string = "{} {} {} {} {} {} {} {} {}".format(python_path, script_path, model, init_train, start_epoch + i*epochs, cycle, epochs, batch_size, save_intervals)

        info_string = "echo {}. cycle, finished: : %date% %time%".format(cycle)
        ##write cmd_string and info_string into batch_file
        batch_file.write(cmd_string)
        batch_file.write("\n")
        batch_file.write(info_string)
        batch_file.write("\n")
        batch_file.write("\n")

    batch_file.write("PAUSE")
    batch_file.close()
    return None

if __name__ == "__main__":
    """
    This script write the train batch file
    """
    write_batch_file(model="VAE_4", start_epoch=0, epochs=250, batch_size=12, save_intervals=250, final_epoch=15000)
