import itertools
import numpy as np
import pandas as pd
import time

from overlappingspheres.functionality import advance
from overlappingspheres.functionality import fractional
from overlappingspheres.functionality import metricpoints
from overlappingspheres.functionality import shift
from typing import List

Vector = List[float]


def perform_experiment(relax: bool, spring_constant: Vector, random_strength: Vector):
    d = []

    iterations = int(1e2)
    # iterations = int(1e4)

    """
    We want to loop through the parameters values
    """

    for iterator in itertools.product(spring_constant, random_strength):

        seconds = time.time()
        test = shift(205)
        for i in range(iterations):
            if i == int(1e1):
                test10 = advance(test, 0.015, "news", i, iterator[0], iterator[1])
            test = advance(test, 0.015, "news", i, iterator[0], iterator[1])
        metric1 = metricpoints(test)
        metric2 = fractional(test)
        m10etric1 = metricpoints(test10)
        m10etric2 = fractional(test10)
        runningtime = (time.time() - seconds)

        filter0 = np.asarray([0])
        filter1 = np.asarray([1])
        filterd0 = test[np.in1d(test[:, 2], filter0)]
        filterd1 = test[np.in1d(test[:, 2], filter1)]
        f10ilterd0 = test10[np.in1d(test10[:, 2], filter0)]
        f10ilterd1 = test10[np.in1d(test10[:, 2], filter1)]

        d.append(
            {
                'relax': relax,
                'spring_constant': iterator[0],
                'random_strength': iterator[1],
                'iterations': iterations,
                'runningtime': runningtime,
                'Distance metric': metric1,
                'Fractional length': metric2,
                '0 points': np.delete(filterd0, np.s_[2:3], axis=1),
                '1 points': np.delete(filterd1, np.s_[2:3], axis=1)
            }
        )

        d.append(
            {
                'relax': relax,
                'spring_constant': iterator[0],
                'random_strength': iterator[1],
                'iterations': 11,
                'runningtime': runningtime,
                'Distance metric': m10etric1,
                'Fractional length': m10etric2,
                '0 points': np.delete(f10ilterd0, np.s_[2:3], axis=1),
                '1 points': np.delete(f10ilterd1, np.s_[2:3], axis=1)
            }
        )

    df1 = pd.DataFrame(d)

    df1.to_csv("df1.csv")
    return df1


print(perform_experiment(False, [25, 50], [5.0, 10.0]))
