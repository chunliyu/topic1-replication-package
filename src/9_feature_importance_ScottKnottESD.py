# https://github.com/klainfo/ScottKnottESD

import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.1.3" # change as needed

from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd


if __name__=='__main__':
    sk = importr('ScottKnottESD')
    data = pd.read_csv("../data/old/data.csv")
    # print(data)
    r_sk = sk.sk_esd(data)

    ranking = pd.DataFrame({'columns': r_sk[2], 'rank': list(r_sk[1])})  # long format
    print("long format: \n")
    print(ranking)

    # ranking = pd.DataFrame([list(r_sk[1])], columns=r_sk[2])  # wide format
    # print("wide format: \n")
    # print(ranking)

    #print("hello")