#!/usr/bin/env python
# coding:utf-8
# @author : liyu

from model import NECS

if __name__ == '__main__':
    data_clusters = {"cornell": 5, "texas": 5, "washington": 5, "wisconsin": 5, "wiki": 17, "cora": 7, "polblogs": 2,
                     "citeseer": 6, "football": 12, "polbooks": 3, "email": 42}
    times = 1
    d = 128
    t = 1
    for name in ['cornell', 'texas', 'washington', 'wisconsin', 'wiki', 'cora', "polblogs", "email", "citeseer",
                 "football", "polbooks"][:1]:
        obj = NECS(name=name)
        for order in [2, 3]:
            for decay in [0, 0.1, 0.2, 0.3]: # decay "0" == order "1"
                for a in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                    for b in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                        print(
                            "Dataset: {}\t"
                            "Dimens: {}\t"
                            "Alpha: {}\t"
                            "Beta: {}\t"
                            "Order:{}\t"
                            "Decay: {}\t"
                            "times: {}\n".format(name, d, a, b, order, decay, t))
                        obj.dimensions = d
                        obj.clusters = data_clusters[name]
                        obj.alpha = a
                        obj.beta = b
                        obj.order = order
                        obj.decay = decay
                        obj.times = t
                        obj.re_matrix_random_initialization()
                        print("H\t{}".format(obj.H.shape))
                        print("U\t{}".format(obj.U.shape))
                        print("W\t{}".format(obj.W.shape))
                        print("V\t{}".format(obj.V.shape))
                        obj.run()
