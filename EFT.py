import torch
import pandas as pd
import pickle
import numpy as np
from transformer.batch import subsequent_mask
import individual_TF
import torch.nn.functional as F
import torch.nn as nn
import datetime
import gc
import matplotlib.pyplot as plt
from Prepare_data import prepare_multiclass_data_from_preprocessed_hdf5

import argparse

#-------------------------------------------------------
# Make sequences of shifted timestamps (x0,x1,x2),(x1,x2,x3),(x2,x3,x4)
def shift(data, slices, step):
    grouped_dataset = []
    for k in data:
        grouped_data = []
        for i in range(k.shape[0] - slices + 1):
            grouped_data.append(k[i * step:i * step + slices])
        grouped_dataset.append(grouped_data)
    return grouped_dataset


# After splitting the data into sequences of the same video, concat them ignoring the video id
def concat_the_sequences(videos):
    seq = []
    for video in videos:
        for sequence in video:
            seq.append(sequence)
    return seq


def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def onehot_initialization(a):
    ncols = 4
    out = np.zeros(a.shape + (ncols,), dtype=float)
    out[all_idx(a, axis=2)] = 1
    return out


def train_epoch(model, x_train, y_train, optim, device, criterion, start_time):
    model.train()
    epoch_loss, n_correct, n_total = 0, 0, 0
    inp = iter(x_train)

    for id_b, batch in enumerate(y_train):
        # optim.zero_grad()
        optim.optimizer.zero_grad()

        # Input features
        src = next(inp).to(device)

        # Target starting point
        start_of_sequence = torch.tensor(np.zeros((1, 4))).repeat(batch.shape[0], 1, 1)
        trg = torch.cat((start_of_sequence, batch), 1)

        # Shift target by one for next token prediction
        target_input = trg[:, :-1, :].to(device)
        target = trg[:, 1:, :].to(device)
        _, target = target.max(2)

        # Make the mask for the next speaker sequences
        src_att = torch.ones((src.shape[0], 1, src.shape[1])).to(device)
        sequence_length = target_input.size(1)
        trg_att = subsequent_mask(sequence_length).repeat(target.shape[0], 1, 1).to(device)

        pred = model(src.float(), target_input.float(), src_att, trg_att)
        pred = torch.transpose(pred, 1, 2)

        # loss
        loss = criterion(pred, target)

        loss.backward()
        optim.step()
        epoch_loss += loss.item()

        pred = torch.transpose(pred, 1, 2)

        # print(pred)
        # print(np.shape(pred))

        # accuracy
        _, predicted = pred.max(2)
        n_correct += predicted.eq(target).sum().item()
        n_total += target.size(0) * target.size(1)

    time_elapsed = datetime.datetime.now() - start_time
    print("Time elapsed:", str(time_elapsed).split(".")[0])

    accuracy = 100. * n_correct / n_total
    loss_per_epoch = epoch_loss / len(x_train)
    return loss_per_epoch, accuracy



def eval_epoch(model, x_val, y_val, device, criterion):
    model.eval()
    epoch_loss, n_correct, n_total = 0, 0, 0
    inp = iter(x_val)
    with torch.no_grad():
        for id_b, batch in enumerate(y_val):
            src = next(inp).to(device)
            # Target starting point
            start_of_sequence = torch.tensor(np.zeros((1, 4))).repeat(batch.shape[0], 1, 1)
            trg = torch.cat((start_of_sequence, batch), 1)

            # Shift target by one for next token prediction
            target_input = trg[:, :-1, :].to(device)
            target = trg[:, 1:, :].to(device)
            _, target = target.max(2)

            # Make the mask for the next speaker sequences
            src_att = torch.ones((src.shape[0], 1, src.shape[1])).to(device)
            sequence_length = target_input.size(1)
            trg_att = subsequent_mask(sequence_length).repeat(target.shape[0], 1, 1).to(device)

            pred = model(src.float(), target_input.float(), src_att, trg_att)
            pred = torch.transpose(pred, 1, 2)

            # loss
            loss = criterion(pred, target)

            epoch_loss += loss.item()

            pred = torch.transpose(pred, 1, 2)

            # accuracy
            _, predicted = pred.max(2)
            n_correct += predicted.eq(target).sum().item()
            n_total += target.size(0) * target.size(1)

    accuracy = 100. * n_correct / n_total
    loss_per_epoch = epoch_loss / len(x_val)
    return loss_per_epoch, accuracy





class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


#-------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='EgoCom Turn-Taking Prediction')
    parser.add_argument('--future-pred', default=5, type=int,
                        help='Specifies on how long in the future we are going'
                             'to predict. Default use of predicting 1 sec in the'
                             'future. Other values to be used are 1,3,5,10')
    parser.add_argument('--history-sec', default=5, type=int,
                        help='Specifies on how many seconds of the past we are'
                             'using as a history feature. Default value is 4sec'
                             'but also can be used 4,10,30.')
    parser.add_argument('--modals', default='text', type=str,
                        help='Specifies which modals will be used.'
                             'Default value is all the modals.'
                             'Other values that can be determined are'
                             '"text","video","voxaudio","text_video_voxaudio",'
                             '"video_voxaudio", "text_voxaudio", "text_video".')
    parser.add_argument('--include-prior', default=None, type=str,
                        help='By default (None) this will run train both a model'
                             'with a prior and a model without a prior.'
                             'Set to "true" to include the label of the current'
                             'speaker when predicting who will be speaking in the'
                             'the future. Set to "false" to not include prior label'
                             'information. You can think of this as a prior on'
                             'the person speaking, since the person who will be'
                             'speaking is highly related to the person who is'
                             'currently speaking.')
    parser.add_argument('--sequence-len', default=10, type=int,
                        help='Determine the sequence length of the transformer.'
                             'Could be any integer and has a default vale 10.')


    # Extract argument flags
    args = parser.parse_args()
    future_pred = args.future_pred
    history_sec = args.history_sec
    modals = args.modals
    sequence_len = args.sequence_len

    if args.include_prior is None:
        include_prior_list = [True, False]
    elif args.include_prior.lower() == 'true':
        include_prior_list = [True]
    elif args.include_prior.lower() == 'false':
        include_prior_list = [False]
    else:
        raise ValueError('--include prior should be None, "true", or "false')

    #-------Assertion the modality
    assert modals in ['text','voxaudio', 'video', 'text_voxaudio', 'text_video', 'video_voxaudio' , 'text_video_voxaudio']

    #--------------------------------
    modals = ['text','voxaudio','video', 'text_voxaudio', 'text_video', 'video_voxaudio' ,'text_video_voxaudio']
    include_prior_list = [True, False ]
    layers = 6
    emb_size=512
    heads=8
    #N=903
    max_epoch = 10
    best_acc = 0.0
    warmup = 10
    dropout = 0.1




    #----------GPU Usage
    device=torch.device("cuda")
    #cpu='store_true'
    # if cpu or not torch.cuda.is_available():
    #    device=torch.device("cpu")



    for modal in modals:
        for include_prior in include_prior_list:
           # del model
            #gc.collect()
            #torch.cuda.empty_cache()
            if not include_prior and modal=='text_video_voxaudio':
                flag_settings = {
                    'future_pred': future_pred,
                    'history_sec': history_sec,
                    'modals': modal, #modals
                    'include_prior': include_prior, #list
                    'sequence_len' : sequence_len,
                }
                print('Running with settings:', flag_settings)


                print("Max epoch is = ", max_epoch)


                x_train,y_train, x_val, y_val, x_test, y_test = prepare_multiclass_data_from_preprocessed_hdf5(
                    modal,
                    history_sec,
                    future_pred,
                    include_prior)



                x_train = concat_the_sequences(shift(x_train,sequence_len,1))
                y_train = concat_the_sequences(shift(y_train,sequence_len,1))
                x_val = concat_the_sequences(shift(x_val,sequence_len,1))
                y_val = concat_the_sequences(shift(y_val,sequence_len,1))
                x_test = concat_the_sequences(shift(x_test,sequence_len,1))
                y_test = concat_the_sequences(shift(y_test,sequence_len,1))



                x_train = np.stack(x_train)
                y_train = onehot_initialization(np.stack(y_train))
                #print(x_train.shape, y_train.shape)
                x_val = np.stack(x_val)
                y_val = onehot_initialization(np.stack(y_val))
                #print(x_val.shape , y_val.shape)
                x_test = np.stack(x_test)
                y_test = onehot_initialization(np.stack(y_test))
                #print(x_test.shape, y_test.shape)

                x_tr_dl = torch.utils.data.DataLoader(x_train,  batch_size=25)
                y_tr_dl = torch.utils.data.DataLoader(y_train,  batch_size=25)
                x_val_dl = torch.utils.data.DataLoader(x_val, batch_size=25)
                y_val_dl = torch.utils.data.DataLoader(y_val, batch_size=25)
                x_test_dl = torch.utils.data.DataLoader(x_test, batch_size=25)
                y_test_dl = torch.utils.data.DataLoader(y_test, batch_size=25)



                N = x_val.shape[2]



                #optim =  torch.optim.SGD(model.parameters(), lr=0.01)
                criterion = nn.CrossEntropyLoss()
                loss_values, loss_val_values, train_acc_values, val_acc_values, test_acc_values = [] , [] , [] , [], []
                start_time = datetime.datetime.now()


                #--------Model
                model=individual_TF.IndividualTF(N, 4, 4, N=layers,
                                                 d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)


                #--------Optimizer
                optim = NoamOpt(512, 1., len(x_tr_dl)*warmup,
                                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


                start_time = datetime.datetime.now()
                for epoch in range(max_epoch):
                    train_loss, train_accuracy = train_epoch(model, x_tr_dl, y_tr_dl,optim,device, criterion, start_time)
                    loss_values.append(train_loss)
                    train_acc_values.append(train_accuracy)
                    print('Epoch: %03i/%03i | Train Loss: %.3f | Accuracy: %.3f'%(epoch+1, max_epoch, train_loss ,train_accuracy))
                    val_loss, val_accuracy = eval_epoch(model, x_val_dl, y_val_dl, device, criterion)
                    loss_val_values.append(val_loss)
                    val_acc_values.append(val_accuracy)
                    print('Epoch: %03i/%03i | Val Loss : %.3f | Val Accuracy %.3f'%(epoch+1, max_epoch, val_loss , val_accuracy))
                    test_loss, test_accuracy = eval_epoch(model, x_test_dl, y_test_dl, device, criterion)
                    test_acc_values.append(test_accuracy)
                    print('Epoch: %03i/%03i | Test Accuracy %.3f'%(epoch+1, max_epoch, test_accuracy))
                    if val_accuracy>best_acc:
                        best_acc = val_accuracy
                        #save_name = str(history_sec)+ '_'+ str(future_pred) + '_' \
                        #+str(include_prior) + '_'+modals +'_'+str(setting['lr'])+'_'+str(setting['dropout'])+'_'+str(setting['weight_decay'])
                        save_name = str(history_sec)+ '_'+ str(future_pred) + '_' \
                        +str(include_prior) + '_'+modal +'_'+'NoamOpt'
                        print("Saving model with test accuracy : ", val_accuracy)
                        with open('Models/Test_version/model_'+ save_name,'wb') as f:
                            pickle.dump(model,f)

                plt.figure()
                plt.plot(loss_values, label = "train loss")
                plt.plot(loss_val_values , label ="val loss")
                leg = plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.savefig('Graphs/Test_Version/Loss_'+ save_name +'.png')
                plt.show()
                plt.figure().clear()
                plt.close()
                plt.cla()
                plt.clf()

                plt.figure()
                plt.plot(train_acc_values, label = "train acc")
                plt.plot(val_acc_values, label = "val acc")
                plt.plot(test_acc_values, label = "test acc")
                leg = plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy %")
                plt.savefig('Graphs/Test_Version/Accuracy_'+ save_name +'.png')
                plt.show()
                plt.figure().clear()
                plt.close()
                plt.cla()
                plt.clf()





if __name__=="__main__":
    main()
                  
