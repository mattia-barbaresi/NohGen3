import json
from datetime import datetime
import os
import random
import utils
import markov


def create(file_name, file_in_sep, random_seed):
    """Generate tps from sequences in file_in"""

    # set random
    random.seed(random_seed)

    # Create target dir if don't exist
    dir_out = "data/models/" + file_name + "_" + str(random_seed) + "/"

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    else:
        print("Directory ", dir_out, "already exists")

    # calculate model and form classes
    ti = datetime.now()
    sequences, voc = utils.read_from_file("data/" + file_name + ".txt", separator=file_in_sep)
    os.mkdir(dir_out + "model/")
    with open(dir_out + "model/alphabet.json", "w") as fp:
        json.dump(voc, fp)
    markov.compute(sequences, dir_name=dir_out + "model/")
    print("Model of " + file_name + " computed... time: ", (datetime.now() - ti).total_seconds(), "s.")
    # plots.plot_tps(dir_out, tps)

    return dir_out


if __name__ == "__main__":
    create("bicinia", " ", 8)
