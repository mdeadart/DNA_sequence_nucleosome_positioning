
from math import floor, ceil, cos, sin, pi, sqrt

import numpy

class DNA_Encoding:

    def GlobalEncoding(self, seq, L=4):

        Feat = []

        H = numpy.zeros((4, len(seq))).astype(int)

        L = L  # size of the patch（补丁） used for extracting the descriptors
        if len(seq) < L:

            print(seq)
            seq[len(seq): L] = seq[len(seq)]  # # to avoid proteins too short

        Len = len(seq)
        for i in range(Len):
            t = 0
            if seq[i] == 'A':
                H[t, i] = 1
            else:
                H[t, i] = 0

            t = t + 1

            if seq[i] == 'C':
                H[t, i] = 1
            else:
                H[t, i] = 0

            t = t + 1

            if seq[i] == 'G':
                H[t, i] = 1
            else:
                H[t, i] = 0

            t = t + 1

            if seq[i] == 'T':
                H[t, i] = 1
            else:
                H[t, i] = 0

            t = t + 1

        F = numpy.zeros((4, L * 2))  # for 0 and 1 fopr each subsequences

        # 揷omposition, " i.e., the frequency of 0s and 1s
        for i in range(4):  # the 10 binarysequence H

            S = max([len(seq) / L, 1])
            t = 0
            for j in range(1, L + 1):

                F[i, t] = round(list(H[i, :])[floor((j - 1) * S): floor(j * S)].count(1) / S, 4)
                t = t + 1
                # F[i, t] = round(list(H[i,:])[ floor((j - 1)  * S): floor((j) * S)-1].count(0)/S,4)

                if j == 1:
                    F[i, t] = round(list(H[i, :])[floor((j - 1) * S): floor((j) * S)].count(0) / S, 4)
                else:
                    F[i, t] = round(list(H[i, :])[floor((j - 1) * S): floor((j) * S) - 1].count(0) / S, 4)
                t = t + 1

        # ransition? i.e., the percent of frequency with which 1 is followed by 0 or 0 is followed by 1 in a characteristic sequence

        F1 = [0] * 4  # for 0-1 transition, 1 , 11, 111 fopr each subsequences

        for i in range(4):  # the 10 binarysequence H
            S = max([len(seq) / L, 1])
            t = 0

            temp = []
            for j in range(1, L + 1):
                Sezione = list(H[i, :])[floor((j - 1) * S): floor((j * S) - 1)]
                Sezione1 = Sezione[1:len(Sezione)]
                Sezione2 = Sezione[2:len(Sezione)]

                # print(Sezione)
                # print(Sezione1)

                counter1 = 0
                counter2 = 0
                counter3 = 0
                for k in range(len(Sezione1)):
                    if Sezione[k] == 1 and Sezione1[k] == 0:
                        counter1 += 1
                    if Sezione[k] == 0 and Sezione1[k] == 1:
                        counter1 += 1

                for k in range(len(Sezione1)):

                    if Sezione[k] == 1 and Sezione1[k] == 1:
                        counter2 += 1

                for k in range(len(Sezione2)):

                    if Sezione[k] == 1 and Sezione1[k] == 1 and Sezione2[k] == 1:
                        counter3 += 1

                # print(counter1)
                # print(counter2)
                # print(counter3)
                temp.extend([counter1, counter2, counter3])

            F1[i] = temp

        F1 = numpy.array(F1).astype("float")

        F = list(F.astype("float").flatten("F"))
        F1 = list(F1.flatten("F"))
        F.extend(F1)

        return numpy.array(F)
