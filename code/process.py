import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import os
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def get_list_of_prefix_from_dir(path):
    list_of_file = os.listdir(path)
    return [f.split(".expr")[0] for f in list_of_file]
    
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
    df = df +1
    df_norm = df.div(df.sum(axis=1), axis=0)
    # Scale up the normalized counts for better numeric stability
    df_norm *= 1e6
    # Apply log transformation (adding a small pseudocount to avoid log(0))
    df_log = np.log1p(df_norm)
    scaler = StandardScaler()
    df_standardized = df_log.T.apply(lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten())
    return df_standardized

def process_to_expression_matrix(name_prefix, dataset_id = '9'):
    file = f'/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/expression_matrix_{dataset_id}celltypes_07072023/{name_prefix}.expr.bed.gz'
    df = read_gzipped_bed(file)
    df.set_index('gene', inplace = True)
    expression_matrix = df.iloc[:,5:]
    return expression_matrix


def process_file(prefix, gene_num = 500, random = False):
    expression_matrix = process_to_expression_matrix(prefix)
    if random:
        out = expression_matrix.loc[get_random_genes(expression_matrix, gene_num)]
    else:
        out = expression_matrix.loc[get_hvg_across_ind(expression_matrix, gene_num)]
    return log_normalize(out)


def process_cov_model_prediction(expression_df):
    X, y = process_cov_and_merge_cov(expression_df)
    ret = fit_model(X, y)
    return ret
    

def process_cov_and_merge_cov(expression_df):
    cov = pd.read_csv('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/PEC2_sample_metadata_processed.csv', sep =',' )
    # Assuming cov is your DataFrame
    cov.set_index('Individual_ID', inplace=True)
    # Let's drop the columns 'Sample_ID' and 'Notes' 
#     cov = cov.drop(columns=['Notes'])
    # Replace '89+' with '90'
    cov['Age_death'] = cov['Age_death'].replace('89+', '90')
    cov['Age_death'] = cov['Age_death'].replace('90+', '90')
    # Convert 'Age_death' to float
    cov['Age_death'] = cov['Age_death'].astype(float)
    # Use pandas get_dummies to one-hot encode the categorical features
    cov_encoded = pd.get_dummies(cov, columns=['Cohort', 'Biological_Sex', 'Disorder', '1000G_ancestry'])
    cov_encoded = cov_encoded.dropna()
    #combine to do the classification
    combined_df = cov_encoded.join(expression_df)
    cov_encoded = combined_df.dropna()
    # Separate features and target variable
    X = cov_encoded.drop('Age_death', axis=1)
    y = cov_encoded['Age_death']
    return X, y
    
def fit_model(X, y, model = 'XGBoost'):
    # baseline
    X = X.iloc[:, :]
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model == 'XGBoost':
        # Initialize and fit the model
        model = xgboost.XGBRegressor(random_state=42).fit(X, y)
        model.fit(X_train, y_train)
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
    return model, X_train, X_test, y_train, y_test

    # Now 'predictions' will hold the predicted 'Age_death' for the test set.
    
def predict(list_of_prefix, gene_num = 500, fixed_index = None):
    if len(list_of_prefix) == 1:
        print(f'=======> Processing {list_of_prefix[0]}')
        expression_df = process_file(list_of_prefix[0], gene_num)
        if fixed_index is not None:
            expression_df = expression_df.loc[fixed_index]
        return process_cov_model_prediction(expression_df)
    for pre in list_of_prefix:
        print(f'=======> Processing {pre}')
        expression_df = process_file(pre, gene_num)
        if fixed_index is not None:
            expression_df = expression_df.loc[fixed_index]
        process_cov_model_prediction(expression_df)
        
def merge_df(list_of_prefix, gene_num = 500):
    expression_df = None
    for pre in list_of_prefix:
#         print(f'=======> Processing {pre}')
        expression_df_curr = process_file(pre, gene_num)
        expression_df_curr.columns = [pre + column for column in expression_df_curr.columns]
        if expression_df is not None:
            expression_df =  pd.merge(expression_df, 
                                      expression_df_curr, 
                                      left_index=True, 
                                      right_index=True, 
                                      how='inner')
        else:
            expression_df = expression_df_curr
    return expression_df


def get_index_intersect(list_of_prefix):
    return merge_df(list_of_prefix, 1).index

def train_val_test_split(X, y, test_size = 0.2, val_size = 0.25, random_state = 42):
    X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val \
    = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state) # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    predict(['SMC'])