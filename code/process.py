import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import os
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

def get_list_of_prefix_from_dir(path, dataset_id = '9'):
    if dataset_id == '9' or dataset_id == '24':
        list_of_file = os.listdir(path)
        return [f.split(".expr")[0] for f in list_of_file]
    elif dataset_id == '21':
        list_of_file = os.listdir(path)
        return [f.split('~')[0] for f in list_of_file]
    else:
        raise NotImplementedError()


def read_gzipped_bed(file_path):
    with gzip.open(file_path, 'rt') as f:
        data = pd.read_csv(f, sep="\t")
    return data

def get_hvg_across_ind(df, num_top_genes = 1000):
    variances = df.var(axis=1)
    hvg = variances.sort_values(ascending=False)
    top_hvg = hvg.head(num_top_genes)
    return top_hvg.index

def get_random_genes(df, gene_num = 1000, seed = 42):
    import numpy as np
    np.random.seed(seed)
    idx = np.random.choice(df.index, gene_num, replace = False)
    return idx


def log_normalize(df):
    # Normalize counts to total counts per cell
    df = df + 1
    df_norm = df.div(df.sum(axis=1), axis=0)
    # Scale up the normalized counts for better numeric stability
    df_norm *= 1e6
    # Apply log transformation (adding a small pseudocount to avoid log(0))
    df_log = np.log1p(df_norm)
    scaler = StandardScaler()
    df_standardized = df_log.T.apply(lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten())
    return df_standardized

def process_to_expression_matrix(name_prefix, dataset_id = '9'):
    if dataset_id == '9' or dataset_id == '24':
        file = f'/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/expression_matrix_{dataset_id}celltypes_07072023/{name_prefix}.expr.bed.gz'
        df = read_gzipped_bed(file)
        df.set_index('gene', inplace = True)
        expression_matrix = df.iloc[:,5:]
    elif dataset_id == '21':
        file = f'/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/expression_matrix/{name_prefix}~expression~matrix~Andy~12072022.txt'
        df = pd.read_csv(file, sep = '\t')
        df.set_index('gid', inplace = True)
        expression_matrix = df.iloc[:,5:]
    return expression_matrix


def process_file(prefix, gene_num = 500, dataset_id = '9', random = False):
    expression_matrix = process_to_expression_matrix(prefix, dataset_id = dataset_id)
    if random:
        out = expression_matrix.loc[get_random_genes(expression_matrix, gene_num)]
    else:
        out = expression_matrix.loc[get_hvg_across_ind(expression_matrix, gene_num)]
    return log_normalize(out)


def process_cov_model_prediction(expression_df, dataset_id = '9', **kwags):
    X, y = process_cov_and_merge_cov(expression_df, dataset_id = dataset_id, **kwags)
    ret = fit_model(X, y, **kwags)
    return ret
    

def process_cov_and_merge_cov(expression_df, dataset_id = '9', **kwags):
    if dataset_id == '9' or dataset_id == '24':
        cov = pd.read_csv('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/PEC2_sample_metadata_processed.csv', sep =',' )
        # Assuming cov is your DataFrame
        cov.set_index('Individual_ID', inplace=True)
    else:
        cov = pd.read_csv('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/covariates/PEC2_all_cohorts_covariates.txt', sep ='\t' )
        # Assuming cov is your DataFrame
        cov.set_index('Sample_ID', inplace=True)
        # Let's drop the columns 'Sample_ID' and 'Notes' 
        cov = cov.drop(columns=['Notes'])
    # Let's drop the columns 'Sample_ID' and 'Notes' 
#     cov = cov.drop(columns=['Notes'])
    # Replace '89+' with '90'
    cov['Age_death'] = cov['Age_death'].replace('89+', '90')
    cov['Age_death'] = cov['Age_death'].replace('90+', '90')
    # Convert 'Age_death' to float
    cov['Age_death'] = cov['Age_death'].astype(float)
    # Use pandas get_dummies to one-hot encode the categorical features
    cov_encoded = pd.get_dummies(cov, 
        columns=['Cohort', 'Biological_Sex', 'Disorder', 
        '1000G_ancestry' if dataset_id in ['9', '24'] else 'Genotype ancestry'])    
    cov_encoded = cov_encoded.dropna()
    #combine to do the classification
    combined_df = cov_encoded.join(expression_df)
    cov_encoded = combined_df.dropna()
    # Separate features and target variable
    X = cov_encoded.drop('Age_death', axis=1)
    y = cov_encoded['Age_death']
    return X, y


def AD_model_split(X, y):
    X_test = X.loc[X['Disorder_Alzheimers/dementia'] == 1]
    y_test = y.loc[X['Disorder_Alzheimers/dementia'] == 1]
    X_train = X.loc[(X['Disorder_Control'] == 1) | (X['Disorder_control'] == 1)]
    y_train = y.loc[(X['Disorder_Control'] == 1) | (X['Disorder_control'] == 1)]
    return X_train, X_test, y_train, y_test


def stratified_train_test_split(X, y, test_size, random_state, num_bins = 10):
    bins = np.linspace(np.min(y), np.max(y), num_bins)

    # Assign each data point to a bin
    y_binned = np.digitize(y, bins)

    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_binned)

    return X_train, X_test, y_train, y_test



def fit_model(X, y, model = 'XGBoost', grid_search_cv = 0, random_state = 42, AD_model = False, only_cov = False, remove_cov = False, stratify = True):
    # Split the dataset into train and test sets
    if not AD_model:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    elif stratify:
        X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = AD_model_split(X, y)

    if only_cov:
        X_train = X_train.iloc[:,:25]
        X_test = X_test.iloc[:,:25]
    elif remove_cov:
        X_train = X_train.iloc[:,25:]
        X_test = X_test.iloc[:,25:]
    else:
        pass

    if model == 'XGBoost':
        if grid_search_cv >0:
            xgb = xgboost.XGBRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'colsample_bytree': [0.3, 0.7],
                'gamma': [0.0, 0.1, 0.2]
            }
            grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv = grid_search_cv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
            # Print the best parameters

            grid.fit(X_train, y_train)
            print("Best parameters found: ", grid.best_params_)
            # Initialize and fit the model
            model = xgboost.XGBRegressor(**grid.best_params_, random_state=random_state)
        else:
            model = xgboost.XGBRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        # Make predictions on the test data
    else:
        raise NotImplementedError()

    # Make predictions
    predictions = model.predict(X_test)

    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    # Assuming y_test are your ground truth values and predictions are your model's predictions
    correlation, _ = pearsonr(y_test, predictions)
    rmse_error = np.sqrt(mean_squared_error(y_test, predictions))
    mae_error = mean_absolute_error(y_test, predictions)
    rho, _ = spearmanr(y_test, predictions)
    print('Pearson correlation: %.3f' % correlation)
    print('Spearman correlation: %.3f' % rho)
    print('RMSE: %.3f' % rmse_error )
    print('MAE: %.3f' % mae_error)
    return model, X_train, X_test, y_train, y_test, predictions, \
    {'Pearson Correlation': correlation, 'Spearman Correlation': rho, 'RMSE': rmse_error, 'MAE': mae_error}

    # Now 'predictions' will hold the predicted 'Age_death' for the test set.
    
def predict(list_of_prefix, gene_num = 500, fixed_index = None, dataset_id = '9', **kwags):
    if len(list_of_prefix) == 1:
        print(f'=======> Processing {list_of_prefix[0]}')
        expression_df = process_file(list_of_prefix[0], gene_num, dataset_id = dataset_id)
        if fixed_index is not None:
            expression_df = expression_df.loc[fixed_index]
        print(f'number of samples {len(expression_df.index)}')
        print(f'number of genes {gene_num}')
        return process_cov_model_prediction(expression_df, dataset_id = dataset_id, **kwags)
    list_of_predictions = []
    list_of_results = []
    for pre in list_of_prefix:
        print(f'=======> Processing {pre}')
        expression_df = process_file(pre, gene_num, dataset_id = dataset_id)
        if fixed_index is not None:
            expression_df = expression_df.loc[fixed_index]
        print(f'number of samples {len(expression_df.index)}')
        print(f'number of genes {gene_num}')
        model, X_train, X_test, y_train, y_test, predictions, ret = process_cov_model_prediction(expression_df, dataset_id = dataset_id, **kwags)
        list_of_predictions.append(predictions)
        ret['Celltype'] = pre
        ret['number of samples'] = len(expression_df.index)
        ret['number of genes'] = gene_num
        list_of_results.append(ret)
    

    # Baseline predictions
    if fixed_index is not None:
        print(f'=======> Processing Baseline')
        model, X_train, X_test, y_train, y_test, predictions, ret = process_cov_model_prediction(expression_df, dataset_id = dataset_id,  
        only_cov = True, **kwags)
        list_of_predictions.append(predictions)
        ret['Celltype'] = 'Baseline'
        ret['number of samples'] = len(expression_df.index)
        ret['number of genes'] = gene_num
        list_of_results.append(ret)
    final_results = pd.DataFrame(list_of_results)
    return list_of_predictions, y_test, final_results

def predict_with_different_seeds(list_of_prefix, gene_num = 500, fixed_index = None, dataset_id = '9', random_state = 42, num_splits = 10, **kwags):
    list_of_seeds = [i + random_state for i in range(num_splits)]
    final_results_list = []
    for seed in list_of_seeds:
        print(f'=======> Running with seed {seed}')
        list_of_predictions, y_test, final_results = predict(list_of_prefix, 
        gene_num = gene_num, 
        fixed_index = fixed_index, 
        dataset_id = dataset_id, 
        random_state = seed)
        final_results_list.append(final_results)

    final_results = pd.concat(final_results_list, ignore_index=True)
    return final_results


def filter_cell_types(list_of_prefix, dataset_id = '9', min_cell_number = 100, portion = None):
    out = []
    total = get_total_number_of_samples(dataset_id)
    for pre in list_of_prefix:
        expression_df = process_file(pre, 1, dataset_id = dataset_id)
        sample_number = len(expression_df.index)
        if sample_number >= min_cell_number:
            if portion is not None:
                if sample_number >= total * portion:
                    out.append(pre)
            else:
                out.append(pre)
    return out


def get_total_number_of_samples(dataset_id):
    if dataset_id == '9' or dataset_id == '24':
        return len(pd.read_csv('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/PEC2_sample_metadata_processed.csv', sep =',' ).index)
    elif dataset_id == '21':
        return len(pd.read_csv('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/covariates/PEC2_all_cohorts_covariates.txt', sep ='\t').index)
    else:
        raise NotImplementedError()

def merge_df(list_of_prefix, gene_num = 500, dataset_id = '9'):
    expression_df = None
    for pre in list_of_prefix:
#         print(f'=======> Processing {pre}')
        expression_df_curr = process_file(pre, gene_num, dataset_id=dataset_id)
        expression_df_curr.columns = [pre + '_' + column for column in expression_df_curr.columns]
        if expression_df is not None:
            expression_df =  pd.merge(expression_df, 
                                      expression_df_curr, 
                                      left_index=True, 
                                      right_index=True, 
                                      how='inner')
        else:
            expression_df = expression_df_curr
    return expression_df

def get_index_intersect(list_of_prefix, dataset_id = '9'):
    return merge_df(list_of_prefix, 1, dataset_id=dataset_id).index

def train_val_test_split(X, y, test_size = 0.2, val_size = 0.25, random_state = 42):
    X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val \
    = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state) # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    intersect_of_all = get_index_intersect(
    [
 'Chandelier.Pvalb',
 'Astro',
 'L5_IT',
 'Vip',
 'L2.3_IT', 'Sst.Sst_Chodl', 'Oligo' , 'L4_IT', 'L6_IT'], dataset_id = '21')

    # list_of_predictions, y_test, ret = predict([
    # 'Chandelier.Pvalb',
    # 'Astro',
    # 'L5_IT',
    # 'Vip',
    # 'L2.3_IT', 'Sst.Sst_Chodl', 'Oligo' , 'L4_IT', 'L6_IT'],
    # gene_num = 500, dataset_id='21', fixed_index = intersect_of_all, AD_model = True)

    rets = predict_with_different_seeds([
    'Chandelier.Pvalb',
    'Astro',
    'L5_IT',
    'Vip',
    'L2.3_IT', 'Sst.Sst_Chodl', 'Oligo' , 'L4_IT', 'L6_IT'],
    gene_num = 500, dataset_id='21', fixed_index = intersect_of_all, AD_model = True)
    print(rets)