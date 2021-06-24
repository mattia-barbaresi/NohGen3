#!/usr/bin/env python

# Set Based Complexity
# NOTE: it works only with files composed of words in column
# E.g.:
#
# abcaccba
# bdcbdcdd
# aaabbbbc


import bz2
import math
import optparse
import sys
import zlib


def compute_sbc(file_name):
    res = []
    count = 0
    with open(file_name, "r") as fp:
        for line in fp:
            res.append(line.replace('\n', '').encode())
            count += 1
    if count > 1:
        sbc = SBC("bz2", "9", res)
        return sbc.compute()
    else:
        print("(SBC, path excluded): " + file_name)
        return -1


def compute_sbc_from_pop(pop):
    pop_list = []
    count = 0
    for ind in pop:
        pop_list.append("".join(ind).encode())
        count += 1
    if count > 1:
        sbc = SBC("bz2", "9", pop_list)
        return sbc.compute()
    else:
        print("(SBC, pop excluded): " + str(pop_list))
        return -1


# class SBC
class SBC(object):

    def __init__(self, compressor, level, data):
        self.data = data
        self.n = len(data)
        if compressor == 'zlib':
            self.compress = zlib.compress
        elif compressor == 'bz2':
            self.compress = bz2.compress
        else:
            sys.stderr.write('invalid compressor\n')
            sys.exit(1)

    def compute_all_kappa(self):
        self.kappa = []
        for x in self.data:
            result = float(len(self.compress(x)))
            l = len(set(x))  # count number of unique chars
            result = result / math.log(l, 2)  # weight with max entropy
            self.kappa.append(result)

    def compute_ncd(self, a, b):
        ca = float(len(self.compress(a)))
        cb = float(len(self.compress(b)))
        cab = float(len(self.compress(a + b)))
        return 1 - (cab - min(ca, cb)) / max(ca, cb)

    def compute_all_ncd(self):
        m = []
        for i in range(self.n):
            maux = []
            for j in range(i, self.n):
                maux.append(self.compute_ncd(self.data[i], self.data[j]))
            m.append(maux)
        self.ncd = m

    def compute_all_effe(self):
        self.xi = 2.0 / (self.n * (self.n - 1))
        self.effe = []
        for i in range(self.n):
            s = 0.0
            # a,b -> m[a][b-a], if a <= b
            #     -> m[b][a-b], if a > b
            for j in range(i + 1, self.n):  # true matrix indices!
                s = s + self.ncd[i][j - i] * (1 - self.ncd[i][j - i])
            for j in range(0, i):  # true matrix indices!
                s = s + self.ncd[j][i - j] * (1 - self.ncd[j][i - j])
            s = s * self.xi
            self.effe.append(s)

    def compute(self):
        self.compute_all_kappa()
        self.compute_all_ncd()
        self.compute_all_effe()
        xi = 1.0 / self.n
        s = 0.0
        for i in range(self.n):
            s = s + self.kappa[i] * self.effe[i]
        s = xi * s
        return s


# main
def main():
    opt = optparse.OptionParser("usage: %prog [OPTION] [FILE]")
    opt.add_option('-c', dest='compressor', default='zlib', choices=('zlib', 'bz2'),
                   help='compressor algorithm (zlib,bz2)')
    opt.add_option('-l', dest='level', default='9', choices=map(str, range(1, 10)), help='compression level (1-9)')
    (options, files) = opt.parse_args()

    data = []
    f = open(files[0])
    f = sys.stdin
    for line in f.readlines():
        data.append(line.rstrip())

    sbc = SBC(options.compressor, options.level, data)

    # print sbc.comp()
    # print sbc.ncd_matrix()
    print(sbc.compute())


########
if __name__ == '__main__':
    main()
