import numpy as np

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    #print G_pred, G_true
    return G_pred/G_true

def cv_split(train_z, labels_z, kfold, kiter):
    train_subsets_k = []
    label_subsets_k = []
    n_sample = train_z.shape[0] / kfold
    data = np.column_stack((train_z, labels_z))
    for k in range(kiter):
        np.random.shuffle(data)
        train = data[:, :-1]
        labels = data[:, -1]
        train_subsets = []
        label_subsets = []
        for i in range(kfold):
            tmp_train = np.array( np.concatenate( (np.copy(train[0:(i*n_sample), :]), np.copy(train[((i+1)*n_sample):, :])), axis=0 ), copy=True )
            tmp_val = np.array(train[(i * n_sample):((i+1)*n_sample), :], copy=True)

            tmp_train_label = np.array( np.concatenate( (np.copy(labels[0:(i*n_sample)]), np.copy(labels[((i+1)*n_sample):])), axis=0 ), copy=True)
            tmp_val_label = np.array(labels[(i*n_sample):((i+1)*n_sample)], copy=True)

            #print 'train', i, tmp_train[1,:10]
            #print 'val', i, tmp_val[1,:10]
            train_subsets.append([ tmp_train, tmp_val ])
            label_subsets.append([ tmp_train_label, tmp_val_label ])
        train_subsets_k.append( train_subsets )
        label_subsets_k.append( label_subsets )
    return train_subsets_k, label_subsets_k


def write_submission(idx, pred, filename):
    preds = pd.DataFrame({"Id": idx, "Hazard": pred})
    preds = preds.set_index("Id")
    preds.to_csv(filename)
