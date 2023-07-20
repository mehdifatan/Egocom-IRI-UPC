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



def train_epoch(model1, model2, model3, x_train, y_train, optim1, optim2, optim3, device, criterion1, criterion2, criterion3, start_time):
    model1.train()
    model2.train()
    model3.train()

    # y_train1=y_train.clone()
    # y_train2=y_train.Clone()
    sfmx = nn.Softmax(dim=1)

    epoch_loss, n_correct, n_total = 0, 0, 0
    inp = iter(x_train)

    for id_b, batch in enumerate(y_train):
        # optim.zero_grad()
        optim1.optimizer.zero_grad()
        optim2.optimizer.zero_grad()
        optim3.optimizer.zero_grad()

        # Input features
        src = next(inp).to(device)

        src2 = src[:, :, :899].clone()
        src3 = src[:, :, 900:7043].clone()
        src4 = src[:, :, 7044:].clone()

        # Target starting point
        start_of_sequence = torch.tensor(np.zeros((1, 4))).repeat(batch.shape[0], 1, 1)
        trg = torch.cat((start_of_sequence, batch), 1)

        # Shift target by one for next token prediction
        target_input = trg[:, :-1, :].to(device)
        target = trg[:, 1:, :].to(device)
        _, target = target.max(2)

        # Make the mask for the next speaker sequences
        src_att1 = torch.ones((src2.shape[0], 1, src2.shape[1])).to(device)
        src_att2 = torch.ones((src3.shape[0], 1, src3.shape[1])).to(device)
        src_att3 = torch.ones((src4.shape[0], 1, src4.shape[1])).to(device)

        sequence_length = target_input.size(1)
        trg_att = subsequent_mask(sequence_length).repeat(target.shape[0], 1, 1).to(device)

        pred1 = model1(src2.float(), target_input.float(), src_att1, trg_att)
        # print("before transpose=",np.shape(pred1))
        pred1 = sfmx(torch.transpose(pred1, 1, 2))

        pred2 = model2(src3.float(), target_input.float(), src_att2, trg_att)
        pred2 = sfmx(torch.transpose(pred2, 1, 2))

        pred3 = model3(src4.float(), target_input.float(), src_att3, trg_att)
        pred3 = sfmx(torch.transpose(pred3, 1, 2))


        # loss
        loss1 = criterion1(pred1, target)
        loss1.backward()
        optim1.step()


        loss2 = criterion2(pred2, target)
        loss2.backward()
        optim2.step()

        loss3 = criterion3(pred3, target)
        loss3.backward()
        optim3.step()



        epoch_loss += (loss1.item() + loss2.item() + loss3.item()) / 3



        pred = (pred1 + pred2 + pred3) / 3

        _, predicted = pred.max(1)

        n_correct += predicted.eq(target).sum().item()
        n_total += target.size(0) * target.size(1)

    time_elapsed = datetime.datetime.now() - start_time
    print("Time elapsed:", str(time_elapsed).split(".")[0])

    accuracy = 100. * n_correct / n_total

    loss_per_epoch = epoch_loss / len(x_train)
    return loss_per_epoch, accuracy



def eval_epoch(model1, model2, model3, x_val, y_val, device, criterion1, criterion2, criterion3):
    model1.eval()
    model2.eval()
    model3.eval()

    epoch_loss, n_correct, n_total = 0, 0, 0
    inp = iter(x_val)
    with torch.no_grad():
        for id_b, batch in enumerate(y_val):
            src = next(inp).to(device)

            src2 = src[:, :, :899].clone()
            src3 = src[:, :, 900:7043].clone()
            src4 = src[:, :, 7044:].clone()

            # Target starting point
            start_of_sequence = torch.tensor(np.zeros((1, 4))).repeat(batch.shape[0], 1, 1)
            trg = torch.cat((start_of_sequence, batch), 1)

            # Shift target by one for next token prediction
            target_input = trg[:, :-1, :].to(device)
            target = trg[:, 1:, :].to(device)
            _, target = target.max(2)

            # Make the mask for the next speaker sequences
            src_att1 = torch.ones((src2.shape[0], 1, src2.shape[1])).to(device)
            src_att2 = torch.ones((src3.shape[0], 1, src3.shape[1])).to(device)
            src_att3 = torch.ones((src4.shape[0], 1, src4.shape[1])).to(device)

            sequence_length = target_input.size(1)
            trg_att = subsequent_mask(sequence_length).repeat(target.shape[0], 1, 1).to(device)

            pred1 = model1(src2.float(), target_input.float(), src_att1, trg_att)
            pred1 = torch.transpose(pred1, 1, 2)

            pred2 = model2(src3.float(), target_input.float(), src_att2, trg_att)
            pred2 = torch.transpose(pred2, 1, 2)

            pred3 = model3(src4.float(), target_input.float(), src_att3, trg_att)
            pred3 = torch.transpose(pred3, 1, 2)



            # loss
            loss1 = criterion1(pred1, target)

            loss2 = criterion2(pred2, target)

            loss3 = criterion3(pred3, target)



            epoch_loss += (loss1.item() + loss2.item() + loss3.item()) / 3

            # accuracy
            pred = (pred1 + pred2 + pred3) / 3
            _, predicted = pred.max(1)

            n_correct += predicted.eq(target).sum().item()
            n_total += target.size(0) * target.size(1)

    accuracy = 100. * n_correct / n_total
    loss_per_epoch = epoch_loss / len(x_val)
    return loss_per_epoch, accuracy




#Make sequences of shifted timestamps (x0,x1,x2),(x1,x2,x3),(x2,x3,x4)
def shift(data, slices ,step):
    grouped_dataset=[]
    for k in data:
        grouped_data=[]
        for i in range(k.shape[0]-slices+1):
            grouped_data.append(k[i*step:i*step+slices])
        grouped_dataset.append(grouped_data)
    return grouped_dataset

#After splitting the data into sequences of the same video, concat them ignoring the video id
def concat_the_sequences(videos):
    seq=[]
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


def main():

    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='EgoCom Turn-Taking Prediction')
    parser.add_argument('--future-pred', default=10, type=int,
                        help='Specifies on how long in the future we are going'
                             'to predict. Default use of predicting 1 sec in the'
                             'future. Other values to be used are 1,3,5,10')
    parser.add_argument('--history-sec', default=5, type=int,
                        help='Specifies on how many seconds of the past we are'
                             'using as a history feature. Default value is 4sec'
                             'but also can be used 4,10,30.')

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
    sequence_len = args.sequence_len


    #args.include_prior='false'

    if args.include_prior is None:
        include_prior=False
    elif args.include_prior.lower() == 'true':
        include_prior = True
    elif args.include_prior.lower() == 'false':
        include_prior= False
    else:
        raise ValueError('--include prior should be None, "true", or "false')


    #--------------------------------
    modal = 'text_video_voxaudio'
    include_prior_list = [False]
    layers = 6
    emb_size=512
    heads=8
    warmup = 10
    dropout = 0.1
    max_epoch=10
    best_acc = 0.0
    N1=899
    N2=6143
    N3=1536

    device=torch.device("cuda:1")
    dropout = 0.1
    #cpu='store_true'
    #if cpu or not torch.cuda.is_available():
    #   device=torch.device("cpu")

    #print(moodel)


    flag_settings = {
        'future_pred': future_pred,
        'history_sec': history_sec,
        'modals': modal, #modals
        'include_prior': include_prior, #list
        'sequence_len' : sequence_len,
    }
    print('Running with settings:', flag_settings)



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


    #optim =  torch.optim.SGD(model.parameters(), lr=0.01)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()

    loss_values, loss_val_values, train_acc_values, val_acc_values, test_acc_values = [] , [] , [] , [], []

    start_time = datetime.datetime.now()




    model1=individual_TF.IndividualTF(N1, 4, 4, N=layers,d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)

    model2=individual_TF.IndividualTF(N2, 4, 4, N=layers,d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)

    model3=individual_TF.IndividualTF(N3, 4, 4, N=layers,d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)




    optim1 = NoamOpt(512, 1., len(x_tr_dl)*warmup,
                        torch.optim.Adam(model1.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    optim2 = NoamOpt(512, 1., len(x_tr_dl)*warmup,
                        torch.optim.Adam(model2.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    optim3 = NoamOpt(512, 1., len(x_tr_dl)*warmup,
                        torch.optim.Adam(model3.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))



    start_time = datetime.datetime.now()
    for epoch in range(max_epoch):
        train_loss, train_accuracy = train_epoch(model1,model2,model3, x_tr_dl, y_tr_dl,optim1,optim2,optim3,device, criterion1, criterion2, criterion3, start_time)
        loss_values.append(train_loss)
        train_acc_values.append(train_accuracy)
        print('Epoch: %03i/%03i | Train Loss: %.3f | Accuracy: %.3f'%(epoch+1, max_epoch, train_loss ,train_accuracy))
        val_loss, val_accuracy = eval_epoch(model1,model2,model3, x_val_dl, y_val_dl, device, criterion1, criterion2, criterion3)
        loss_val_values.append(val_loss)
        val_acc_values.append(val_accuracy)
        print('Epoch: %03i/%03i | Val Loss : %.3f | Val Accuracy %.3f'%(epoch+1, max_epoch, val_loss , val_accuracy))
        test_loss,test_accuracy = eval_epoch(model1,model2,model3, x_test_dl, y_test_dl, device, criterion1, criterion2, criterion3)
        test_acc_values.append(test_accuracy)
        print('Epoch: %03i/%03i | Test Accuracy %.3f'%(epoch+1, max_epoch, test_accuracy))


        if val_accuracy>best_acc:
            best_acc = val_accuracy

            print("The best test accuracy so far is: ",test_accuracy)

            #save_name = str(history_sec)+ '_'+ str(future_pred) + '_' \
            #+str(include_prior) + '_'+modals +'_'+str(setting['lr'])+'_'+str(setting['dropout'])+'_'+str(setting['weight_decay'])
            #save_name = str(history_sec)+ '_'+ str(future_pred) + '_' \
            #+str(include_prior) + '_'+modal +'_'+str(test_accuracy)+'NoamOpt'
            #print("Saving model with test accuracy : ", val_accuracy)
            #with open('Models/Test_version/model1_'+ save_name,'wb') as f:
            #    pickle.dump(model1,f)


            #save_name = str(history_sec)+ '_'+ str(future_pred) + '_' \
            #+str(include_prior) + '_'+modal +'_'+str(test_accuracy)+'NoamOpt'
            #print("Saving model with test accuracy : ", val_accuracy)
            #with open('Models/Test_version/model2_'+ save_name,'wb') as f:
            #    pickle.dump(model2,f)


            #save_name = str(history_sec)+ '_'+ str(future_pred) + '_' \
            #+str(include_prior) + '_'+modal +'_'+str(test_accuracy)+'NoamOpt'
            #print("Saving model with test accuracy : ", val_accuracy)
            #with open('Models/Test_version/model3_'+ save_name,'wb') as f:
            #    pickle.dump(model3,f)




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