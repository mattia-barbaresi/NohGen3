"""
Runs multiple GAs on params values
"""
from datetime import datetime
import generate_models
from main import run_ga
import multiprocessing as mp


def _apply_fun(x):
    # fname, ranseed, novmeth
    run_ga(x[0], x[1], x[2])


def main():
    data = []
    pool = mp.Pool(mp.cpu_count()-1)
    start_time = datetime.now()

    seeds = [7]
    methods = ["multi_log_genotype"]
    files = [
        # {"name": "input", "sep": ""},
        # {"name": "input2", "sep": ""},
        {"name": "irish", "sep": " "},
        {"name": "bicinia", "sep": " "},
        {"name": "all_irish-notes_and_durations-abc", "sep": " "},
        # {"name": "all_songs_in_G", "sep": ""}, # generated only for seed = 7
    ]

    # file name and separator
    for fl in files:
        # seed for random
        for rs in seeds:
            # generate markov model
            generate_models.create(fl["name"], fl["sep"], rs)

            for nov_method in methods:
                data.append([fl["name"], rs, nov_method])
    #
    # # multiprocessing
    pool.map(_apply_fun, data)
    pool.close()
    pool.join()
    print("batch time elapsed :", (datetime.now() - start_time).total_seconds(), "sec.")


if __name__ == "__main__":
    main()
