"""The intelligence layer: features -> prediction -> utility -> decision.

    feature_engineering.py  live network state -> a feature vector
    outcomes.py             counterfactual evaluator (training truth +
                            analytical fallback)
    dataset.py              manufacture labelled training data
    train.py                fit and save the XGBoost models
    predict.py              inference, with graceful fallback
    utility.py              score the trade-off, explainably
    agent.py                the decision-maker, with policy and safety

XGBoost is the only ML model used here. Deep learning and reinforcement
learning remain out of scope, as does multicast.

The agent plugs into the simulator through the same `mode_selector` seam
the fixed rules use, so nothing in the network model needed to change to
accommodate it.
"""
