import docplex.mp.model as cpx
import pandas as pd
import numpy as np
import sys, os
import json

###################################################################################################

options = pd.read_csv("data/filtered_tsla_options_w_greeks.csv")
options = options[options.date_current == "2020-07-13"]
options = options[options.days_to_expiry.isin([95.0, 186.0, 249.0])]

strikes = options.strike_price.value_counts()
strikes = strikes[strikes >= 3].index
options = options[options.strike_price.isin(strikes)]

###################################################################################################

## Params
N = len(options)
MAX_POSITION = 8
N_TRANSCATIONS = 8

## Variables
delta = np.concatenate((options.delta.values, -1 * options.delta.values))
gamma = np.concatenate((options.gamma.values, -1 * options.gamma.values))
vega = np.concatenate((options.vega.values, -1 * options.vega.values))
vomma = np.concatenate((options.vomma.values, -1 * options.vomma.values))
ultima = np.concatenate((options.ultima.values, -1 * options.ultima.values))

## Model
opt_model = cpx.Model(name="VolCrush Model")

## Variables
x_vars = {
    i : opt_model.integer_var(name=f"l{i}", lb=0, ub=MAX_POSITION)
    for i in range(N)
}
x_vars.update({
    i+len(options) : opt_model.integer_var(name=f"s{i}", lb=0, ub=MAX_POSITION)
    for i in range(N)
})

## Constraints
constraints = {
    i : opt_model.add_constraint(
        ct = opt_model.sum(x_vars[i] + x_vars[i + N]) <= MAX_POSITION,
        ctname = f"position_constraint_{i}"
    )
    for i in range(N)
}
constraints[len(constraints)] = opt_model.add_constraint(
    ct = opt_model.sum(x_vars[i] for i in range(2 * N)) == N_TRANSCATIONS,
    ctname = f"minimum_position_constraint"
)
constraints[len(constraints)] = opt_model.add_constraint(
    ct = opt_model.sum(x_vars[i] for i in range(2 * N)) >= 1,
    ctname = f"minimum_position_constraint"
)

## Objective
objective = opt_model.sum(
    
    opt_model.abs(
        opt_model.sum(
            x_vars[i] * delta[i]
            for i in range(2 * N)
        )
    )
    +
    opt_model.abs(
        opt_model.sum(
            x_vars[i] * gamma[i]
            for i in range(2 * N)
        )
    )
    +
    opt_model.abs(
        opt_model.sum(
            x_vars[i] * vega[i]
            for i in range(2 * N)
        )
    )
    +
    opt_model.sum(
        x_vars[i] * vomma[i]
        for i in range(2 * N)
    )
    +
    opt_model.sum(
        x_vars[i] * ultima[i]
        for i in range(2 * N)
    )
    
)
opt_model.minimize(objective)

if __name__ == '__main__':

    opt_model.solve(log_output=True)
    solution = opt_model.solution
    
    n_results = len(os.listdir("cplex_results"))
    with open(f"cplex_results/results_{n_results}.json", "w") as file:
        file.write(json.dumps(solution.export_as_json_string()))

