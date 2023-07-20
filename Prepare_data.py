import torch
import pandas as pd
import pickle
import itertools
import numpy as np

# Location where dataset and pre-processed data is stored.
egocom_loc = "/home/DATASETS/egocom_dataset/Dataset/egocom_pretrained_features/features_by_history_future/" 


# Make sure GPU can be used with PyTorch
device = torch.device('cuda')
# # Data preparation
video_info = pd.read_csv("/home/mehdi/EgoCom-Dataset/Transformer-Egocom-main/video_info.csv")
kinds = ['text', 'video', 'voxaudio'] # ['audio', 'text', 'video', 'voxaudio']

with open(egocom_loc + "feature_column_names.p", 'rb') as rf: #allagi to mesaio
    cols = pickle.load(rf)
# Generate all combinations of "kinds" of features (i.e. modalities).
experiments = list(
    itertools.chain.from_iterable(itertools.combinations(kinds, r)
                                  for r in range(len(kinds)+1))
)[1:]
experiments = {
    "_".join(e): [c for c in cols if c.split("_")[0] in [z+"feat" for z in e]]
    for e in experiments
}
multiclass_prior_feature = 'prior_multiclass_speaker_label'
label_shifts = [0, 2, 4, 9]
histories = [4, 5, 10, 30] 

def datapreprocess(conv_id,data):
    data = data.loc[data['conversation_id']== conv_id]
    x_train = (data[data['train']][0])
    y_train = data[data['train']]['multiclass_speaker_label'].values
    x_val = (data[data['val']][0])
    y_val = data[data['val']]['multiclass_speaker_label'].values
    x_test = (data[data['test']][0])
    y_test = data[data['test']]['multiclass_speaker_label'].values
    data = x_train, y_train, x_val, y_val, x_test, y_test
    return data


def prepare_multiclass_data_from_preprocessed_hdf5(
    experiment_key,
    history,
    future,
    include_prior,
):
    """Produce X_train, X_test, Y_train, Y_test from a preprocessed
    hdf5 file storing the data. Data is already z-score normalized,
    per-column.
    Use this when prediction_task == 'multi' """

    prediction_task = 'multi'
    hdf5_fn = 'egocom_feature_data_normalized_history_{}_future_{}_binary' \
              '.hdf5'.format(history, future)
    experiment = experiments[experiment_key]
    new_data = pd.read_hdf(egocom_loc + hdf5_fn, key=hdf5_fn)
    new_data.dropna(inplace=True)  # Remove NaN values if they exist.
    # Include prior features if part of this experiment
    if include_prior:
        experiment += [multiclass_prior_feature]
        x = new_data[multiclass_prior_feature]  # Z-score normalize prior
        new_data[multiclass_prior_feature] = (x - x.mean()) / x.std()
    # Only use 3 speaker conversations because we are going to combine the
    # features from all 3 in multi-class setting and we need all examples
    # to have the same number of features.
    new_data = new_data[new_data['num_speakers'] == 3]

    # Combine all three speakers to single input for each conversation
    gb_cols = ["conversation_id", "clip_id", "multiclass_speaker_label",
               "test", "train", "val"]
    X = new_data.groupby(gb_cols).apply(
        lambda x: x[experiment].values.flatten()).reset_index()

    # Only include examples with all three speakers (same dimension)
    # this occurs at the end of a video (if one speaker's video is ~1s
    # longer than the others) where there are features for that
    # speaker but not the other speakers.
    input_length = max([len(z) for z in X[0]])
    mask = [len(z) == input_length for z in X[0]]
    X = X[mask]

    dataset_x, dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y = [], [], [], [], [],[] 

    for i in np.unique(X['conversation_id']):
        #print(i)
        
        data_x,data_y, val_data_x, val_data_y, test_data_x, test_data_y = datapreprocess(i,X)
        if len(data_x)!=0:
            data_x = np.stack(data_x)
            dataset_x.append(data_x)
            dataset_y.append(data_y)
        elif len(val_data_x)!=0:
            val_data_x = np.stack(val_data_x)
            val_dataset_x.append(val_data_x)
            val_dataset_y.append(val_data_y)
        elif len(test_data_x)!=0:
            test_data_x = np.stack(test_data_x)
            test_dataset_x.append(test_data_x)
            test_dataset_y.append(test_data_y)
    
    return dataset_x, dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y
