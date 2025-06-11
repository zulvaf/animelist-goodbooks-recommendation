import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from collections import OrderedDict, defaultdict
import itertools

def compute_ranking_predictions(
    algo,
    data,
    usercol,
    itemcol,
    predcol,
    remove_seen=False,
):
    """Computes predictions of an algorithm from Surprise on all users and items in data. It can be used for computing
    ranking metrics like NDCG.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pandas.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    preds_lst = []
    users = data[usercol].unique()
    items = data[itemcol].unique()

    for user in tqdm(users):
        for item in items:
            preds_lst.append([user, item, algo.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions

def ranking_eval(
    model,
    train_set,
    test_set,
    exclude_unknowns=True,
    mode="last",
    verbose=False,
):

    rankings = []
    scores = []
    user_sessions = defaultdict(list)
    session_ids = []
    for [sid], [mapped_ids], [session_items] in tqdm(
        test_set.si_iter(batch_size=1, shuffle=False),
        total=len(test_set.sessions)):

        # if len(session_items) < 2:  # exclude all session with size smaller than 2
        #     continue
        user_idx = test_set.uir_tuple[0][mapped_ids[0]]
        session_ids.append(sid)

        start_pos = 1 if mode == "next" else len(session_items) - 1
        for test_pos in range(start_pos, len(session_items), 1):
            test_pos_items = session_items[test_pos]

            # binary mask for ground-truth positive items
            u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
            u_gt_pos_mask[test_pos_items] = 1

            # binary mask for ground-truth negative items, removing all positive items
            u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
            u_gt_neg_mask[test_pos_items] = 0

            # filter items being considered for evaluation
            if exclude_unknowns:
                u_gt_pos_mask = u_gt_pos_mask[: train_set.num_items]
                u_gt_neg_mask = u_gt_neg_mask[: train_set.num_items]

            u_gt_pos_items = np.nonzero(u_gt_pos_mask)[0]
            u_gt_neg_items = np.nonzero(u_gt_neg_mask)[0]
            item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]


            item_rank, item_scores = model.rank(
                user_idx,
                item_indices,
                history_items=session_items[:test_pos],
                history_mapped_ids=mapped_ids[:test_pos],
                sessions=test_set.sessions,
                session_indices=test_set.session_indices,
                extra_data=test_set.extra_data,
            )
            item_scores = item_scores[item_rank]
            item_rank = [key for value in item_rank for key, val in train_set.iid_map.items() if val == value]

            rankings.append(item_rank)
            scores.append(item_scores)

    return rankings, scores