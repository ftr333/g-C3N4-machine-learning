import time
import json
import pickle
import pathlib
import numpy as np
import pandas as pd


from sklearn import metrics
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor,ExtraTreesRegressor



def get_scores(model,X_train,y_train,X_test,y_test, X_val=None, y_val=None):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    maes = metrics.mean_absolute_error(y_test,y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test,y_pred)
    print(f'Test scores: MAE={maes:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}')
    if (X_val is not None) and (y_val is not None):
        y_pred = model.predict(X_val)
        maes_val = metrics.mean_absolute_error(y_val,y_pred)
        mse_val = metrics.mean_squared_error(y_val,y_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = metrics.r2_score(y_val,y_pred)
        print(f'Val scores: MAE={maes_val:.3f}, RMSE={rmse_val:.3f}, R2={r2_val:.3f}')
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        return maes, rmse, r2, maes_val, rmse_val, r2_val
    else:
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')
        return maes, rmse, r2


def reformat_results(folder):
    def fill_None(test_scores):
        if isinstance(test_scores, dict):
            null_keys = [i for i in test_scores.keys() if len(test_scores[i]) == 0]
            nonnull_keys = [i for i in test_scores.keys() if len(test_scores[i]) != 0]
            test_scores = pd.DataFrame({i: test_scores[i] for i in nonnull_keys})
            for key in null_keys:
                test_scores[key] = None
        return test_scores

    if pathlib.Path(f'{folder}/all_dat.pkl').is_file():
        with open(f'{folder}/all_dat.pkl', 'rb') as f:
            [size_old_val, ids, test_scores, val_scores] = pickle.load(f)
    elif pathlib.Path(f'{folder}/all_dat.pkl.tmp').is_file():
        with open(f'{folder}/all_dat.pkl.tmp', 'rb') as f:
            [size_old_val, ids, test_scores, val_scores] = pickle.load(f)

    else:
        with open(f'{folder}/val_scores.pkl', 'rb') as f:
            val_scores = pickle.load(f)

        with open(f'{folder}/test_scores.pkl', 'rb') as f:
            test_scores = pickle.load(f)

        with open(f'{folder}/ids.pkl', 'rb') as f:
            ids = pickle.load(f)

        with open(f'{folder}/size_old_val.pkl', 'rb') as f:
            size_old_val = pickle.load(f)

    dat_size = pd.DataFrame(size_old_val)
    dat_size.columns = ['val_size']
    tot_train_val_size = len(ids['train_val'])
    dat_size['val_ratio'] = dat_size['val_size'] / tot_train_val_size
    dat_size['train_size'] = tot_train_val_size - dat_size['val_size']
    dat_size['train_ratio'] = 1 - dat_size['val_ratio']
    test_scores['train_ratio'] = dat_size['train_ratio']  # .drop_duplicates()
    val_scores['train_ratio'] = dat_size['train_ratio']  # .drop_duplicates()

    # -- Fix a mini bug in my previous code ----
    diff = len(test_scores['train_ratio']) - len(test_scores['r2'])
    if diff > 0:
        test_scores['train_ratio'] = test_scores['train_ratio'][:-diff]
        # with open(f'{folder}/all_dat.pkl'+'.tmp','wb') as f:
        #     pickle.dump([size_old_val,ids,test_scores,val_scores],f)

    diff = len(val_scores['train_ratio']) - len(val_scores['r2'])
    if diff > 0:
        val_scores['train_ratio'] = val_scores['train_ratio'][:-diff]
    #     with open(f'{folder}/all_dat.pkl'+'.tmp','wb') as f:
    #         pickle.dump([size_old_val,ids,test_scores,val_scores],f)

    # ------------------------------------------

    test_scores = fill_None(test_scores)
    val_scores = fill_None(val_scores)

    ids['train_val'] = (ids['old_val'] + ids['train_new_val'])
    ids['train_val'].reverse()

    with open(f'{folder}/all_dat.pkl', 'wb') as f:
        pickle.dump([size_old_val, ids, test_scores, val_scores], f)

def get_data(
        data_path = 'data_filled_400.csv',
        test_size=0.1,
        random_state=0,
        target='NO(%)',
        fixed_train_ids=None,
        get_pruned_set=False,
        pruning_model=None,
        pruned_set_min_frac=0.3,
        col_X=None,
        standardized=True,
        data_dir='data',
        predefined_train_val_test=False,
        ):
    if col_X is None:
        df = pd.read_csv(data_path)
        df = df.drop(columns=['element1', 'element2'],axis=1)

    # Get standardized X, and y
    if col_X is None:
        X_no_std = df.iloc[:,:-1]
        if standardized:
            X = pd.DataFrame(
                StandardScaler().fit_transform(X_no_std),
                index=X_no_std.index, columns=X_no_std.columns
            )
        else:
            X = X_no_std
    else:
        X = df[col_X]

    y = df[target]
    # 是否选择修剪设置，pruned_set_min_frac,设置为选择训练集的前百分之多少，默认为30%
    if get_pruned_set:
        guiding = pruning_model
        folder = f'{target}/{target}_{guiding}_guiding_pruning'
        reformat_results(folder)
        with open(f'{folder}/all_dat.pkl', 'rb') as f:
            [size_old_val, ids, test_scores, val_scores] = pickle.load(f)
        n_data = int(pruned_set_min_frac * len(ids['train_val']))
        ids_to_use = ids['train_val'][:n_data]
        X = X.loc[ids_to_use]
        y = y.loc[ids_to_use]

    # Random train-val-test split
    X_val, y_val = None, None
    if fixed_train_ids is None:
        if predefined_train_val_test:
            # load json file
            with open(f'{data_dir}/train_val_test.json', 'r') as f:
                train_val_test = json.load(f)
                train = list(train_val_test["train"].keys())
                val = list(train_val_test["val"].keys())
                # val = []
                test = list(train_val_test["test"].keys())
                # # check if val equal to test
                # if set(val) == set(test):
                #     print('Warning: val and test are the same')
                #     print('Setting val to empty')
                #     val = []

            X_pool = X.loc[train]
            y_pool = y.loc[train]
            X_val = X.loc[val]
            y_val = y.loc[val]
            X_test = X.loc[test]
            y_test = y.loc[test]
            print(f'Size of the training set: {X_pool.shape[0]}')
            print(f'Size of the validation set: {X_val.shape[0]}')
            print(f'Size of the test set: {X_test.shape[0]}')
        else:
            X_pool, X_test, y_pool, y_test = train_test_split(
                X, y, test_size=test_size,random_state=random_state
            )
    else:

        X_fixed_train = X.loc[fixed_train_ids]
        y_fixed_train = y.loc[fixed_train_ids]
        X_pool, X_test, y_pool, y_test = train_test_split(
            X.drop(fixed_train_ids), y.drop(fixed_train_ids),
            test_size=test_size, random_state=random_state
        )
        print(f'Size of the fixed training set: {X_fixed_train.shape[0]}')
        print(f'Size of the pool: {X.drop(fixed_train_ids).shape[0]}')

    if fixed_train_ids is None:
        if X_val is None:
            return df, X, y, X_pool, X_test, y_pool, y_test
        else:
            return df, X, y, X_pool, X_test, y_pool, y_test, X_val, y_val

    else:
        return df, X, y, X_pool, X_test, y_pool, y_test, X_fixed_train, y_fixed_train


def prune_rd(
        model,X_pool,y_pool,X_test,y_test,threshold,train_size,file_out,
        min_drop=None,max_iter=None,drop_max_err=None,model_test=None,
        join_model=False,threshold2=None,
        X_fixed_train=None, y_fixed_train=None,
        retrain=True,
        autorestart=True
        # threshold_take_back = None
           ):

    i_initial = 0
    remove = {}
    ids = {}
    ids['train_val'] = X_pool.index.tolist()  
    ids['old_val'] = []  
    size_old_val = []
    ids_to_remove = []  
    test_scores = {'maes': [], 'rmse': [], 'r2': [], 'maes_m2': [], 'rmse_m2': [], 'r2_m2': []}
    val_scores = {'maes': [], 'rmse': [], 'r2': [], 'maes_m2': [], 'rmse_m2': [], 'r2_m2': []}
    test_fit = {'maes': [], 'rmse': [], 'r2': [], 'maes_m2': [], 'rmse_m2': [], 'r2_m2': []}
 

    if pathlib.Path(file_out + '.tmp').is_file() and autorestart:
        print(file_out + '.tmp' + ' found.')
        with open(file_out + '.tmp', 'rb') as f:
            [size_old_val, ids, test_scores_in, val_scores_in] = pickle.load(f)

        if isinstance(test_scores_in, pd.DataFrame):
            for col in test_scores_in.columns:
                test_scores[col] = test_scores_in[col].tolist()

        if isinstance(val_scores_in, pd.DataFrame):
            for col in val_scores_in.columns:
                val_scores[col] = val_scores_in[col].tolist()

        print('。')
        i_initial = len(test_scores['r2'])

        X = pd.concat([X_pool, X_test])
        y = pd.concat([y_pool, y_test])
        X_pool = X.loc[ids['train_val']]
        y_pool = y.loc[ids['train_val']]
        test_ids = list(set(X.index.tolist()) - set(ids['train_val']))
        X_test = X.loc[test_ids]
        y_test = y.loc[test_ids]
        print(':')
        print(f'X_pool shape: {X_pool.shape}')
        print(f'X_test shape: {X_test.shape}')


    if min_drop is None:
        min_drop = int(X_pool.shape[0] / 1000)

    if max_iter is None:
        max_iter = 10

    for i in range(i_initial, max_iter):
        # Save a temp file
        if i > i_initial + 1:
            with open(file_out + '.tmp', 'wb') as f:
                pickle.dump([size_old_val, ids, test_scores, val_scores], f)

        ids['old_val'].extend(ids_to_remove)
        ids['train_new_val'] = list(set(ids['train_val']) - set(ids['old_val']))
        remove[i] = ids['train_new_val']
        X_train_new_val = X_pool.loc[ids['train_new_val']]
        y_train_new_val = y_pool.loc[ids['train_new_val']]
        # Split
        X_train, X_new_val, y_train, y_new_val = train_test_split(
            X_train_new_val, y_train_new_val, train_size=train_size, random_state=0
        )
        start_time = time.time()

        # Train
        if X_fixed_train is None:
            model.fit(X_train,y_train)
        else:
            model.fit(
                pd.concat([X_train,X_fixed_train]),
                pd.concat([y_train,y_fixed_train])
                )
        # predict and get the abs errors,.sort_values()
        y_err_new_val = (model.predict(X_new_val) - y_new_val).abs().sort_values()
        # drop by threshold，y_err_new_val<threshold
        ids_to_remove = y_err_new_val[y_err_new_val<threshold].index.tolist()

        y_pred = model.predict(X_new_val)
        maes = metrics.mean_absolute_error(y_new_val, y_pred)
        mse = metrics.mean_squared_error(y_new_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_new_val, y_pred)
        test_fit['maes'].append(maes)
        test_fit['rmse'].append(rmse)
        test_fit['r2'].append(r2)
        if join_model:

            if X_fixed_train is None:
                model_test.fit(X_train,y_train)
            else:
                model_test.fit(
                    pd.concat([X_train,X_fixed_train]),
                    pd.concat([y_train,y_fixed_train])
                )
            y_err_new_val = (model_test.predict(X_new_val) - y_new_val).abs().sort_values()
            ids_to_remove = list(
                set(ids_to_remove) & set(y_err_new_val[y_err_new_val<threshold2].index.tolist())
                )

        if min_drop is not None:
            if len(ids_to_remove)  < min_drop:
                ids_to_remove = y_err_new_val.iloc[:min_drop].index.tolist()

        if drop_max_err is not None:
            ids_to_remove.extend(y_err_new_val.iloc[-drop_max_err:].index.tolist())

        size_train_new_val = len(ids['train_new_val'])
        size_to_remove = len(ids_to_remove)
        size_all = X_pool.shape[0]


        if size_to_remove > size_train_new_val+120:
            with open(file_out,'wb') as f:
                pickle.dump([size_old_val,ids,test_scores,val_scores],f)
            print(f'Stopping pruning. size_to_remove={size_to_remove},'+
                  f'min_drop={min_drop}, size_train_new_val={size_train_new_val}')
            return size_old_val,ids,test_scores,val_scores,test_fit,remove

        size_old_val.append(len(ids['old_val']))
        print('================================')
        print(f"Iteration {i}:")
        print(f'==== : {size_to_remove} ')
        print(f'=== old_val count: {size_old_val[-1]} (ratio = {size_old_val[-1]/size_all:.3f})')
        print(f'ratio: {size_train_new_val}  (ratio = {size_train_new_val/size_all:.3f})')
        print("--- %s seconds ---" % (time.time() - start_time))


        start_time = time.time()
        # test scores
        if retrain:
            if X_fixed_train is None:
                model.fit(X_train_new_val, y_train_new_val)
            else:
                model.fit(
                    pd.concat([X_train_new_val, X_fixed_train]),
                    pd.concat([y_train_new_val, y_fixed_train])
                )

        y_pred = model.predict(X_test)
        maes = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)
        print('================：')
        print(f'Test scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
        test_scores['maes'].append(maes)
        test_scores['rmse'].append(rmse)
        test_scores['r2'].append(r2)




        # val scores
        if i > 0:
            y_pred = model.predict(X_pool.loc[ids['old_val']])
            y_old_val = y_pool.loc[ids['old_val']]
            maes = metrics.mean_absolute_error(y_old_val, y_pred)
            mse = metrics.mean_squared_error(y_old_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_old_val, y_pred)
            print(f'Val scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
            val_scores['maes'].append(maes)
            val_scores['rmse'].append(rmse)
            val_scores['r2'].append(r2)
        print("--- %s seconds ---" % (time.time() - start_time))

        '''
        Get test and val scores for the 2nd model (used to check transferability)
        '''
        if model_test is not None:
            # test scores
            start_time = time.time()
            if X_fixed_train is None:
                model_test.fit(X_train_new_val, y_train_new_val)
            else:
                model_test.fit(
                    pd.concat([X_train_new_val, X_fixed_train]),
                    pd.concat([y_train_new_val, y_fixed_train])
                )


            y_pred = model_test.predict(X_new_val)
            maes = metrics.mean_absolute_error(y_new_val, y_pred)
            mse = metrics.mean_squared_error(y_new_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_new_val, y_pred)
            test_fit['maes_m2'].append(maes)
            test_fit['rmse_m2'].append(rmse)
            test_fit['r2_m2'].append(r2)


            y_pred = model_test.predict(X_test)
            maes = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)
            print('==================')
            print(f'Test scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
            test_scores['maes_m2'].append(maes)
            test_scores['rmse_m2'].append(rmse)
            test_scores['r2_m2'].append(r2)

            if i > 0:
                y_pred = model_test.predict(X_pool.loc[ids['old_val']])
                y_old_val = y_pool.loc[ids['old_val']]
                maes = metrics.mean_absolute_error(y_old_val, y_pred)
                mse = metrics.mean_squared_error(y_old_val, y_pred)
                rmse = np.sqrt(mse)
                r2 = metrics.r2_score(y_old_val, y_pred)
                print(f'Val scores: maes={maes:.3f}, rmse={rmse:.3f}, r2={r2:.3f}')
                val_scores['maes_m2'].append(maes)
                val_scores['rmse_m2'].append(rmse)
                val_scores['r2_m2'].append(r2)
            print("--- %s seconds ---" % (time.time() - start_time))



    with open(file_out, 'wb') as f:
        pickle.dump([size_old_val, ids, test_scores, val_scores], f)
    return size_old_val, ids, test_scores, val_scores,test_fit,remove



def return_model(modelname, random_state):


    model = {}
    if modelname == 'XG':
        return XGBRegressor(colsample_bytree=1, eta=0.05, eval_metric='rmse',
                  max_depth=3, n_estimators=400, seed=2, subsample=0.5)
    elif modelname == 'RF':
        return RandomForestRegressor()
    elif modelname == 'CB':
        return CatBoostRegressor(l2_leaf_reg= 3,learning_rate=0.05,
                                 depth=5,iterations = 900,verbose=False)


    elif modelname == 'GB':
        return GradientBoostingRegressor(learning_rate=0.05, max_depth=7, min_samples_leaf=1,
                                         min_samples_split=3,n_estimators=175, subsample=0.7,
                                         random_state=14)
