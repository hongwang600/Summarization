# pylint: skip-file
import sys
import os
import argparse
from subprocess import Popen, PIPE


def run_config(ROUGE_PATH, SUMM_PATH, MODEL_PATH):
    i2summ = {}
    summ2i = {}

    # for result
    flist = os.listdir(SUMM_PATH)
    i = 0
    for fname in flist:
        i2summ[str(i)] = fname
        summ2i[fname] = str(i)
        i += 1

    # for models
    flist = os.listdir(MODEL_PATH)
    i2model = {}
    for fname in flist:
        if fname not in summ2i:
            raise IOError

        i = summ2i[fname]
        i2model[i] = fname

    assert len(i2model) == len(i2summ)

    # write to config file
    rouge_s = "<ROUGE-EVAL version=\"1.0\">"
    for file_id, fsumm in i2summ.items():
        rouge_s += "\n<EVAL ID=\"" + file_id + "\">" \
                   + "\n<PEER-ROOT>" \
                   + SUMM_PATH \
                   + "\n</PEER-ROOT>" \
                   + "\n<MODEL-ROOT>" \
                   + "\n" + MODEL_PATH \
                   + "\n</MODEL-ROOT>" \
                   + "\n<INPUT-FORMAT TYPE=\"SPL\">" \
                   + "\n</INPUT-FORMAT>" \
                   + "\n<PEERS>" \
                   + "\n<P ID=\"C\">" + fsumm + "</P>" \
                   + "\n</PEERS>" \
                   + "\n<MODELS>"

        rouge_s += "\n<M ID=\"" + file_id + "\">" + i2model[file_id] + "</M>"
        rouge_s += "\n</MODELS>\n</EVAL>"

    rouge_s += "\n</ROUGE-EVAL>"

    with open(os.path.join(ROUGE_PATH, "myROUGE_Config.xml"), "w") as f_rouge:
        f_rouge.write(rouge_s)


def run_rouge(ROUGE_PATH, verbose=False):
    os.chdir("./ROUGE-1.5.5")
    perl_command = ["perl", "ROUGE-1.5.5.pl",
                    "-n", "2", "-w", "1.2", "-m", "-2", "4", "-u", "-c", "95", "-r", "1000",
                    "-f", "A", "-p", "0.5", "-t", "0",
                    os.path.join(ROUGE_PATH, "myROUGE_Config.xml"),
                    "C"]
    process = Popen(perl_command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, error = process.communicate()
    os.chdir("..")

    rouge = {
        "ROUGE-1": [0, 0, 0],  # [R, P, F]
        "ROUGE-2": [0, 0, 0],
        "ROUGE-L": [0, 0, 0],
    }
    output = output.decode("utf-8").strip().split("\n")
    for line in output:
        if verbose:
            print(line)
        tmp = line.split()
        if len(tmp) > 3 and tmp[1] in rouge:
            if tmp[2] == "Average_R:":
                rouge[tmp[1]][0] = float(tmp[3])
            elif tmp[2] == "Average_P:":
                rouge[tmp[1]][1] = float(tmp[3])
            elif tmp[2] == "Average_F:":
                rouge[tmp[1]][2] = float(tmp[3])
    return rouge


def run(DIR, verbose=False):
    print("Calculating rouge score of {}".format(DIR))
    ROUGE_PATH = os.path.abspath(DIR)
    SUMM_PATH = os.path.abspath(os.path.join(DIR, "summary"))
    MODEL_PATH = os.path.abspath(os.path.join(DIR, "reference"))

    run_config(ROUGE_PATH, SUMM_PATH, MODEL_PATH)
    return run_rouge(ROUGE_PATH, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--res_dir", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    res = run(args.res_dir, args.verbose)
    for k, v in res.items():
        print("{}: {}".format(k, v))
