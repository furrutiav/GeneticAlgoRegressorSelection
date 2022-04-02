import pandas as pd
import numpy as np
import pickle
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
import random

from sklearn.feature_selection import chi2, SelectKBest

def get_random_stratified_block_selector(size_natural_partition, num_blocks=4):
    stratified_block_selector = np.zeros(size_natural_partition, dtype=int)
    current_set = set(range(size_natural_partition))
    for i in range(num_blocks):
        i_comb = random.sample(current_set,
                               size_natural_partition // num_blocks)  # np.random.choice(current_set, size_natural_partition//num_blocks)
        stratified_block_selector[list(i_comb)] = i
        current_set = current_set.difference(set(i_comb))
    return stratified_block_selector


def get_nb_fold(set_all_indexs, dic_natural_partition, keys_natural_partition, stratified_block_selector):
    num_blocks = max(stratified_block_selector) + 1
    blocks = []
    for i in range(num_blocks):
        stratified_block_test = np.where(stratified_block_selector==i)[0]
        test_block = set()
        for k in stratified_block_test:
            test_block = test_block.union(set(dic_natural_partition[keys_natural_partition[k]]))
        train_block = set_all_indexs.difference(test_block)
        blocks.append({"test": list(test_block), "train": list(train_block)})
    return blocks


def get_R2_nb_fold(X, y, nb_fold, cols_selector, name_cols):
    selected_cols = name_cols[np.where(cols_selector==1)[0]]
    R2_per_block = []
    for block in nb_fold:
        ix_train, ix_test = block["train"], block["test"]
        X_train, X_test = X.loc[ix_train], X.loc[ix_test]
        y_train, y_test = y.loc[ix_train], y.loc[ix_test]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        model = LinearRegression()
        model.fit(X_train[selected_cols], y_train)

        p = 1
        X_ = [X_train, X_test][p]
        y_ = [y_train, y_test][p]
        y_hat = model.predict(X_[selected_cols])
        R2_score = model.score(X_[selected_cols], y_)
        if R2_score>0:
            R2_per_block.append(R2_score)
    return R2_per_block


# stop = True
# while stop:
#   mean_R2_per_ix_jx_col = [[current_R2, -1, ""]]
#   for ix_col in np.where(current_cols_selector == 1)[0]:
#     for jx_col in np.where(current_cols_selector == 0)[0]:
#       col_ix = name_cols[ix_col]
#       col_jx = name_cols[jx_col]
#       new_cols_selector_ix_jx = current_cols_selector.copy()
#       new_cols_selector_ix_jx[ix_col] = 0
#       new_cols_selector_ix_jx[jx_col] = 1
#       new_R2_nb_fold = get_R2_nb_fold(X, y, nb_fold, new_cols_selector_ix_jx, name_cols)
#       mean_R2_per_ix_jx_col.append([np.mean(new_R2_nb_fold), (ix_col, jx_col), (col_ix, col_jx)])
#   max_R2 = max(mean_R2_per_ix_jx_col)
#   if max_R2[0] > current_R2:
#     current_R2 = max_R2[0]
#     current_cols_selector[max_R2[1][0]] = 0
#     current_cols_selector[max_R2[1][1]] = 1
#     print("Replace (delete/insert):", max_R2[2], current_R2)
#   else:
#     stop = False

def insert_mutation(current_cols_selector, R2, X, y, nb_fold, name_cols, pmf_cols):
    current_R2 = R2
    stop = True
    while stop:
        mean_R2_per_ix_col = [[current_R2, -1, len(nb_fold), ""]]
        add_list = np.array(np.where(current_cols_selector==0)[0], dtype=int)
        if len(add_list) > 20:
            sub_pmf_cols = pmf_cols[add_list]
            sub_pmf_cols = sub_pmf_cols / sum(sub_pmf_cols)
            add_list = np.random.choice(add_list, size=20, replace=False, p=sub_pmf_cols)
        for ix_col in add_list:
            col_ix = name_cols[ix_col]
            new_cols_selector_ix = current_cols_selector.copy()
            new_cols_selector_ix[ix_col] = 1
            new_R2_nb_fold = get_R2_nb_fold(X, y, nb_fold, new_cols_selector_ix, name_cols)
            mean_R2_per_ix_col.append([np.mean(new_R2_nb_fold), ix_col, len(new_R2_nb_fold), col_ix])
            print(pmf_cols[ix_col], mean_R2_per_ix_col[-1])
        max_R2 = max(mean_R2_per_ix_col)
        if (max_R2[0] > current_R2) and (max_R2[2]==len(nb_fold)):
            current_R2 = max_R2[0]
            current_cols_selector[max_R2[1]] = 1
            print("Insert:", max_R2[3], current_R2)
        else:
            stop = False
    return (current_cols_selector, current_R2)


def delete_mutation(current_cols_selector, R2, X, y, nb_fold, name_cols, pmf_cols):
    current_R2 = R2
    stop = True
    while stop:
        mean_R2_per_ix_col = [[current_R2, -1, len(nb_fold), ""]]
        remove_list = np.array(np.where(current_cols_selector==1)[0], dtype=int)
        if len(remove_list) > 20:
            sub_pmf_cols = 1 - pmf_cols[remove_list]
            sub_pmf_cols = sub_pmf_cols / sum(sub_pmf_cols)
            remove_list = np.random.choice(remove_list, size=20, replace=False, p=sub_pmf_cols)
        for ix_col in remove_list:
            col_ix = name_cols[ix_col]
            new_cols_selector_ix = current_cols_selector.copy()
            new_cols_selector_ix[ix_col] = 0
            new_R2_nb_fold = get_R2_nb_fold(X, y, nb_fold, new_cols_selector_ix, name_cols)
            mean_R2_per_ix_col.append([np.mean(new_R2_nb_fold), ix_col, len(new_R2_nb_fold), col_ix])
            print(pmf_cols[ix_col], mean_R2_per_ix_col[-1])
        max_R2 = max(mean_R2_per_ix_col)
        if (max_R2[0] > current_R2) and (max_R2[2] == len(nb_fold)):
            current_R2 = max_R2[0]
            current_cols_selector[max_R2[1]] = 0
            print("Delete:", max_R2[3], current_R2)
        else:
            stop = False
    return (current_cols_selector, current_R2)


def get_mutation(cols_selector, R2, X, y, nb_fold, name_cols, pmf_cols):
    insert_stop = True
    delete_stop = True
    while insert_stop or delete_stop:
        for k, type_mutation in enumerate([insert_mutation, delete_mutation]):
            if [insert_stop, delete_stop][k]:
                print(type_mutation.__name__)
                new_cols_selector, new_R2 = type_mutation(cols_selector.copy(), R2, X, y, nb_fold, name_cols, pmf_cols)
                if k==0:
                    insert_stop = ~np.isclose(new_R2, R2)
                elif k==1:
                    delete_stop = ~np.isclose(new_R2, R2)
                cols_selector, R2 = new_cols_selector, new_R2
    return cols_selector, R2


def get_cross_over(cols_selector_1, cols_selector_2, p=0.7):
    cols_selector_intersection = cols_selector_1 * cols_selector_2
    cols_selector_cross_over = cols_selector_intersection.copy()
    for i in range(2):
        for ix_col in np.where((cols_selector_intersection==0) & ([cols_selector_1, cols_selector_2][i]==1))[0]:
            if np.random.uniform() < [p, 1 - p][i]:
                cols_selector_cross_over[ix_col] = 1
    return cols_selector_cross_over


def get_R2_nb_fold_lasso(X, y, nb_fold, cols_selector, name_cols, alpha_min=0, alpha_max=1.5, alpha_size=25):
    selected_cols = name_cols[np.where(cols_selector==1)[0]]
    alpha_grid = list(np.linspace(alpha_min, alpha_max, alpha_size))
    R2_per_block = []
    R2_per_alpha = [[] for _ in alpha_grid]
    for block in nb_fold:
        ix_train, ix_test = block["train"], block["test"]
        X_train, X_test = X.loc[ix_train], X.loc[ix_test]
        y_train, y_test = y.loc[ix_train], y.loc[ix_test]


        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        block_R2_per_alpha = []
        for ix_alpha, alpha in enumerate(alpha_grid):
            if alpha==0:
                model = LinearRegression()
            else:
                model = Lasso(alpha=alpha)
            model.fit(X_train[selected_cols], y_train)

            p = 1
            X_ = [X_train, X_test][p]
            y_ = [y_train, y_test][p]
            y_hat = model.predict(X_[selected_cols])
            block_R2_per_alpha.append(model.score(X_[selected_cols], y_))
            R2_per_alpha[ix_alpha].append(block_R2_per_alpha[-1])
        R2_per_block.append(max(block_R2_per_alpha))
    return R2_per_block, R2_per_alpha


def get_evaluation_seed(size_natural_partition, num_blocks, set_all_indexs, dic_natural_partition,
                        keys_natural_partition, num_evaluation=25):
    evaluation_stratified_block_selector = [get_random_stratified_block_selector(size_natural_partition, num_blocks) for
                                            _ in range(num_evaluation)]
    list_e_nb_fold = []
    for e, e_stratified_block_selector in enumerate(evaluation_stratified_block_selector):
        e_nb_fold = get_nb_fold(set_all_indexs, dic_natural_partition, keys_natural_partition,
                                e_stratified_block_selector)
        list_e_nb_fold.append(e_nb_fold)
    return evaluation_stratified_block_selector, list_e_nb_fold


def get_evaluation_metrics(list_e_nb_fold, cols_selector, X, y, name_cols, m="lasso"):
    R2_per_e = []
    R2_per_alpha = []
    for e, e_nb_fold in enumerate(list_e_nb_fold):
        if m == "lasso":
            e_R2_per_block, e_R2_per_alpha = get_R2_nb_fold_lasso(X, y, e_nb_fold, cols_selector, name_cols)
            R2_per_alpha.append(np.mean(e_R2_per_alpha, axis=1))
            R2_per_e.append([e, [np.mean(e_R2_per_block), np.mean(e_R2_per_alpha[0])]])
        else:
            e_R2_per_block = np.array(get_R2_nb_fold(X, y, e_nb_fold, cols_selector, name_cols))
            e_R2_per_block = e_R2_per_block[e_R2_per_block>0]
            R2_per_e.append([e, [np.mean(e_R2_per_block), np.mean(e_R2_per_block)]])
            R2_per_alpha.append(np.array(R2_per_e[-1][1]))
    R2_generation = np.max(np.mean(R2_per_alpha, axis=0))
    return R2_generation, R2_per_e


def get_candidates(R2_per_e):
    sorted_R2_per_e = sorted(R2_per_e, key=lambda x: x[1][0], reverse=True)
    o = {
        "best": sorted_R2_per_e.pop(0),
        "worst": sorted_R2_per_e.pop(-1),
        1: sorted_R2_per_e.pop(np.random.choice(range(len(sorted_R2_per_e))))
    }
    return o


def get_candidates_mutation(dic_candidates, cols_selector, X, y, list_e_nb_fold, name_cols, pmf_cols):
    o = {}
    for k, v in dic_candidates.items():
        print(k)
        o[k] = get_mutation(cols_selector, v[1][1], X, y, list_e_nb_fold[v[0]], name_cols, pmf_cols)
    return o


def get_mutation_cross_over(candidates_mutation, cols_selector):
    o = {}
    for k, v in candidates_mutation.items():
        o[f"0/{k}"] = get_cross_over(cols_selector, v[0])
    o["best/worst"] = get_cross_over(candidates_mutation["best"][0], candidates_mutation["worst"][0])
    o["1/worst"] = get_cross_over(candidates_mutation[1][0], candidates_mutation["worst"][0])
    o["best/1"] = get_cross_over(candidates_mutation["best"][0], candidates_mutation[1][0])
    return o


def get_mutation_cross_over_evaluation_metrics(mutation_cross_over, list_e_nb_fold, X, y, name_cols, m="lasso"):
    o = {}
    for k, v in mutation_cross_over.items():
        print(k)
        o[k] = get_evaluation_metrics(list_e_nb_fold, v, X, y, name_cols, m=m)
        print(o[k][0])
    return sorted(o.items(), key=lambda x: x[1][0], reverse=True)


def get_pmf_cols_CLF(X, y):
    dic_label_count = y.value_counts().to_dict()
    max_class = max([v, k] for k, v in dic_label_count.items())[1]
    index_label_1 = y[y==((max_class+1)%2)].index
    oversampling_steps = int(dic_label_count[max_class] / dic_label_count[(max_class+1)%2]) - 1
    X_res, y_res = X.copy(), y.copy()
    for step in range(oversampling_steps):
        new_indexs = [f"{ix}+{step + 1}" for ix in index_label_1]
        copied_sub_X = pd.DataFrame(X.loc[index_label_1].values, columns=X.columns, index=new_indexs)
        copied_sub_y = pd.Series(y.loc[index_label_1].values, index=new_indexs)
        X_res = pd.concat([X_res, copied_sub_X], axis=0)
        y_res = pd.concat([y_res, copied_sub_y], axis=0)

    X_res = pd.DataFrame(StandardScaler().fit_transform(X_res), columns=X_res.columns, index=X_res.index)

    selector = SelectKBest(chi2, k=X.shape[1])
    X_res_ = X_res - X_res.min()
    selector.fit(X_res_, y_res)
    scores_selector = {col: selector.scores_[i] if str(selector.scores_[i]) != "nan" else 0 for i, col in enumerate(X.columns.tolist())}
    pmf_cols = np.array([1.7 ** np.log(scores_selector[c]) for c in X.columns])
    print(pmf_cols)
    pmf_cols = pmf_cols + np.min(pmf_cols)
    pmf_cols = pmf_cols / sum(pmf_cols)
    return pmf_cols


def get_pmf_cols(X, y):
    X_res = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    X_res["label"] = y.loc[X_res.index]
    scores_selector = X_res.corr()["label"].to_dict()
    # pmf_cols = np.array([0.12*np.log(abs(scores_selector[c]))for c in X.columns])
    pmf_cols = np.array([abs(scores_selector[c]) for c in X.columns])
    pmf_cols[np.isnan(pmf_cols)] = np.min(pmf_cols[~np.isnan(pmf_cols)])
    print(list(pmf_cols))
    print(sum(pmf_cols))
    pmf_cols = pmf_cols - np.min(pmf_cols)
    pmf_cols = pmf_cols / sum(pmf_cols)
    return pmf_cols


# def get_greedy_initial_step(X, y, name_cols, pmf_cols, set_all_indexs, dic_natural_partition, keys_natural_partition,
#                             size_natural_partition):
#     sorted_pmf_cols = np.array(sorted(enumerate(pmf_cols), key=lambda x: x[1], reverse=True))
#     ixs_sorted = np.array(sorted_pmf_cols[:, 0], dtype=int)
#     stratified_block_selector = get_random_stratified_block_selector(size_natural_partition, num_blocks)
#     nb_fold = get_nb_fold(set_all_indexs, dic_natural_partition, keys_natural_partition, stratified_block_selector)
#     grid_R2 = []
#     a, b = num_cols // 2, num_cols // 10
#     a = num_cols if a == 0 else a
#     b = num_cols if b == 0 else b
#     print([name_cols[ix] for ix in ixs_sorted][:15])
#     print(a, b)
#     for pivot in np.linspace(1, 100, num=10, dtype=int):
#         cols_selector = np.zeros(num_cols, dtype=int)
#         cols_selector[ixs_sorted[:pivot]] = 1
#         mean_R2 = np.mean(get_R2_nb_fold(X, y, nb_fold, cols_selector, name_cols))
#         grid_R2.append([mean_R2, pivot])
#         print(grid_R2[-1])
#     max_R2 = max(grid_R2)
#     cols_selector = np.zeros(num_cols, dtype=int)
#     cols_selector[ixs_sorted[:max_R2[1]]] = 1
#     return cols_selector, max_R2[0]




