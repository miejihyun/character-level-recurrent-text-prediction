import nltk, math, os, torch

import matplotlib.pyplot as plt
from collections import namedtuple
from nltk.corpus import brown
import numpy as np

import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils import *
from models import *

def init_brown():
    nltk.download('brown')
    words = brown.words(categories=(brown.categories()))
    N     = len(words)

    words_test  = words[0:math.ceil(N*0.7)]
    words_train = words[math.ceil(N*0.7):math.ceil(N*0.9)]
    words_val   = words[math.ceil(N*0.9):]

    f=open('test.txt','w')
    for w in words_test:
        f.write(w+" ")
    f.close()

    f=open('train.txt','w')
    for w in words_train:
        f.write(w+" ")
    f.close()

    f=open('valid.txt','w')
    for w in words_val:
        f.write(w+" ")
    f.close()

num_char = 0
def preprocess(): 
    global num_char
    word_dict, char_dict = create_word_char_dict("valid.txt", "train.txt", "test.txt")
    num_words = len(word_dict)
    num_char  = len(char_dict)
    char_dict["BOW"] = num_char+1
    char_dict["EOW"] = num_char+2
    char_dict["PAD"] = 0
    
    #  dict of (int, string)
    reverse_word_dict = {value:key for key, value in word_dict.items()}
    max_word_len = max([len(word) for word in word_dict])

    objects = {
        "word_dict": word_dict,
        "char_dict": char_dict,
        "reverse_word_dict": reverse_word_dict,
        "max_word_len": max_word_len
    }
    
    torch.save(objects, "cache/prep_"+corpus+".pt")
    print("Preprocess done.")

def dataset_prep():
    train_text = read_data("train.txt")
    valid_text = read_data("valid.txt")
    test_text  = read_data("test.txt")

    train_set = np.array(text2vec(train_text, char_dict, max_word_len))
    valid_set = np.array(text2vec(valid_text, char_dict, max_word_len))
    test_set  = np.array(text2vec(test_text,  char_dict, max_word_len))

    # Labels are next-word index in word_dict with the same length as inputs
    train_label = np.array([word_dict[w] for w in train_text[1:]] + [word_dict[train_text[-1]]])
    valid_label = np.array([word_dict[w] for w in valid_text[1:]] + [word_dict[valid_text[-1]]])
    test_label  = np.array([word_dict[w] for w in test_text[1:]] + [word_dict[test_text[-1]]])

    category = {"tdata":train_set, "vdata":valid_set, "test": test_set, 
                "trlabel":train_label, "vlabel":valid_label, "tlabel":test_label}
    torch.save(category, "cache/data_sets_"+corpus+".pt") 

def train_LSTM(net, data, opt, model_name):
    global results
    torch.manual_seed(1024)

    train_input = torch.from_numpy(data.train_input)
    train_label = torch.from_numpy(data.train_label)
    valid_input = torch.from_numpy(data.valid_input)
    valid_label = torch.from_numpy(data.valid_label)

    L = opt.seq_len

    # [num_seq, seq_len, max_word_len+2]
    num_seq = train_input.size()[0] //  L
    train_input = train_input[:num_seq* L, :]
    train_input = train_input.view(-1,  L, opt.max_word_len+2)

    num_seq = valid_input.size()[0] //  L
    valid_input = valid_input[:num_seq* L, :]
    valid_input = valid_input.view(-1,  L, opt.max_word_len+2)

    num_epoch = opt.epochs
    num_iter_train = train_input.size()[0] // opt.batch_size
    
    learning_rate = opt.init_lr
    old_PPL = 100000
    best_PPL = 100000

    n_stuck = 0

    # Log-SoftMax
    criterion = nn.CrossEntropyLoss()
    
    # word_emb_dim == hidden_size / num of hidden units 
    hidden = (to_var(torch.zeros(2,  opt.batch_size, opt.word_embed_dim)), 
              to_var(torch.zeros(2,  opt.batch_size, opt.word_embed_dim)))

    for epoch in range(num_epoch):
        ################  Validation  ####################
        net.eval()
        loss_batch = []
        PPL_batch  = []
        num_iter_valid = valid_input.size()[0] // opt.batch_size


        # TRANSFORMER :
        # VALID  : S+L sequences
        # LABELS : S+L labels of next words -> use the (-L,1) labels
        
        valid_generator  = batch_generator(valid_input, opt.batch_size)
        vlabel_generator = batch_generator(valid_label, opt.batch_size*L)

        for t in range(num_iter_valid):
            batch_input = valid_generator.__next__()  # (N,L,W)
            batch_label = vlabel_generator.__next__() # (N*L)

            ####################################################################

            hidden = [state.detach() for state in hidden]
            valid_output, hidden = net(to_var(batch_input), hidden)
            ####################################################################

            length = valid_output.size()[0]

            # [num_sample-1, len(word_dict)] vs [num_sample-1]

            valid_loss = criterion(valid_output, to_var(batch_label))

            PPL = torch.exp(valid_loss.data)

            loss_batch.append(float(valid_loss))
            PPL_batch.append(float(PPL))

        PPL = np.mean(PPL_batch)
        print("[epoch {}] valid PPL={}".format(epoch, PPL))
        print("valid loss={}".format(np.mean(loss_batch)))
        print("PPL decrease={}".format(float(old_PPL - PPL)))

        # UPDATE RESULTS DICT
        if results.get(model_name) == None:
            results[model_name] = {"validation":{"loss":[],"PPL":[]},"training":{"loss":[],"PPL":[]}}
        results[model_name]["validation"]["loss"].append(np.mean(loss_batch))
        results[model_name]["validation"]["PPL"].append(PPL)

        # Preserve the best model
        if best_PPL > PPL:
            best_PPL = PPL
            torch.save(net.state_dict(), "cache/"+model_name+".pt")
            torch.save(net, "cache/"+model_name+"_net.pkl")

        # Adjust the learning rate
        if float(old_PPL - PPL) <= 1.0 and n_stuck > 5:
            learning_rate /= 10
            print("halved lr:{}".format(learning_rate))
            n_stuck = 0
        elif float(old_PPL - PPL) <= 1.0:
            n_stuck += 1

        old_PPL = PPL

        ##################################################
        #################### Training ####################
        net.train()

        optimizer  = optim.Adam(net.parameters(), 
                               lr = learning_rate)

        # split the first dim
        input_generator = batch_generator(train_input, opt.batch_size)
        label_generator = batch_generator(train_label, opt.batch_size*L)

        total_loss, total_PPL = 0, 0

        for t in range(num_iter_train):

            if t % 100 == 0:
                print(f"{t}/{num_iter_train}")

            batch_input = input_generator.__next__()
            batch_label = label_generator.__next__()
            ####################################################################

            # detach hidden state of LSTM from last batch
            hidden = [state.detach() for state in hidden]
            
            ####################################################################

            output, hidden = net(to_var(batch_input), hidden)
            # [num_word, vocab_size]
            ####################################################################
            loss = criterion(output, to_var(batch_label))

            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5, norm_type=2)
            optimizer.step()

            total_loss += loss.cpu().data
            total_PPL  += np.exp(loss.cpu().data)
          
        total_loss /= num_iter_train 
        total_PPL  /= num_iter_train 
    
        results[model_name]["training"]["loss"].append(total_loss)
        results[model_name]["training"]["PPL"].append(total_PPL)

        print(f"Train loss : {total_loss}")
        print(f"Train PPL : {total_PPL}")

    torch.save(net.state_dict(), "cache/"+model_name+".pt")
    print("Training finished.")

def train_Transformer(net, data, opt, model_name):
    global results
    torch.manual_seed(1024)

    train_input = torch.from_numpy(data.train_input)
    train_label = torch.from_numpy(data.train_label)
    valid_input = torch.from_numpy(data.valid_input)
    valid_label = torch.from_numpy(data.valid_label)

    L = opt.seq_len

    # [num_seq, seq_len, max_word_len+2]
    num_seq = train_input.size()[0] //  L
    train_input = train_input[:num_seq* L, :]
    train_input = train_input.view(-1,  L, opt.max_word_len+2)

    num_seq = valid_input.size()[0] //  L
    valid_input = valid_input[:num_seq* L, :]
    valid_input = valid_input.view(-1,  L, opt.max_word_len+2)

    num_epoch = opt.epochs
    num_iter_train = train_input.size()[0] // opt.batch_size
    
    learning_rate = opt.init_lr
    old_PPL = 100000
    best_PPL = 100000

    n_stuck = 0

    # Log-SoftMax
    criterion = nn.CrossEntropyLoss()
    
    X = torch.zeros((opt.batch_size,L,opt.max_word_len+2)).long().cuda()

    for epoch in range(num_epoch):
        ################  Validation  ####################
        net.eval()
        loss_batch = []
        PPL_batch  = []
        num_iter_valid = valid_input.size()[0] // opt.batch_size


        # TRANSFORMER :
        # VALID  : S+L sequences
        # LABELS : S+L labels of next words -> use the (-L,1) labels
        
        valid_generator  = batch_generator(valid_input, opt.batch_size)
        vlabel_generator = batch_generator(valid_label, opt.batch_size*L)

        for t in range(num_iter_valid):
            batch_input = valid_generator.__next__()  # (N,L,W)
            batch_label = vlabel_generator.__next__() # (N*L)

            ####################################################################
            y_input  = batch_input[:,:,:].cuda()
            batch_label = torch.reshape(batch_label,(opt.batch_size,L)).cuda()
            y_pred      = batch_label[:,:].cuda().reshape(-1)
            
            ####################################################################
            valid_output = net(X,y_input)

            ####################################################################

            ####
            X = y_input
            ####

            length = valid_output.size()[0]

            # [num_sample-1, len(word_dict)] vs [num_sample-1]
            valid_loss = criterion(valid_output, to_var(y_pred))

            PPL = torch.exp(valid_loss.data)

            loss_batch.append(float(valid_loss))
            PPL_batch.append(float(PPL))

        PPL = np.mean(PPL_batch)
        print("[epoch {}] valid PPL={}".format(epoch, PPL))
        print("valid loss={}".format(np.mean(loss_batch)))
        print("PPL decrease={}".format(float(old_PPL - PPL)))

        # UPDATE RESULTS DICT
        if results.get(model_name) == None:
            results[model_name] = {"validation":{"loss":[],"PPL":[]},"training":{"loss":[],"PPL":[]}}
        results[model_name]["validation"]["loss"].append(np.mean(loss_batch))
        results[model_name]["validation"]["PPL"].append(PPL)

        # Preserve the best model
        if best_PPL > PPL:
            best_PPL = PPL
            torch.save(net.state_dict(), "cache/"+model_name+".pt")
            torch.save(net, "cache/"+model_name+"_net.pkl")

        # Adjust the learning rate
        if float(old_PPL - PPL) <= 1.0 and n_stuck > 5:
            learning_rate /= 10
            print("halved lr:{}".format(learning_rate))
            n_stuck = 0
        elif float(old_PPL - PPL) <= 1.0:
            n_stuck += 1

        old_PPL = PPL

        ##################################################
        #################### Training ####################
        net.train()

        X = torch.zeros((opt.batch_size,L,opt.max_word_len+2)).long().cuda()

        optimizer  = optim.Adam(net.parameters(), 
                               lr = learning_rate)

        # split the first dim
        input_generator = batch_generator(train_input, opt.batch_size)
        label_generator = batch_generator(train_label, opt.batch_size*L)

        total_loss, total_PPL = 0, 0

        for t in range(num_iter_train):

            if t % 100 == 0:
                print(f"{t}/{num_iter_train}")

            batch_input = input_generator.__next__()
            batch_label = label_generator.__next__()

            ####################################################################

            y_input  = batch_input[:,:,:].cuda()
            batch_label = torch.reshape(batch_label,(opt.batch_size,L)).cuda()
            y_pred      = batch_label[:,:].cuda().reshape(-1)

            ####################################################################
            output = net(X,y_input)
            # [num_word, vocab_size]
            ####################################################################


            loss = criterion(output, to_var(y_pred))
  
            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5, norm_type=2)
            optimizer.step()

            ####
            X = y_input
            ####

            total_loss += loss.cpu().data
            total_PPL  += np.exp(loss.cpu().data)
          
        total_loss /= num_iter_train
        total_PPL  /= num_iter_train
    
        results[model_name]["training"]["loss"].append(total_loss)
        results[model_name]["training"]["PPL"].append(total_PPL)

        print(f"Train loss : {total_loss}")
        print(f"Train PPL : {total_PPL}")

    torch.save(net.state_dict(), "cache/"+model_name+".pt")
    print("Training finished.")

def test_LSTM(net, data, opt, model_name):
    net.eval()

    test_input = torch.from_numpy(data.test_input)
    test_label = torch.from_numpy(data.test_label)

    L = opt.seq_len
    num_seq = test_input.size()[0] // opt.seq_len
    test_input = test_input[:num_seq*opt.seq_len, :]
    # [num_seq, seq_len, max_word_len+2]
    test_input = test_input.view(-1, opt.seq_len, opt.max_word_len+2)

    criterion = nn.CrossEntropyLoss()

    loss_list = []
    num_iter_test = test_input.size()[0] // opt.batch_size
    test_generator = batch_generator(test_input, opt.batch_size)
    label_generator = batch_generator(test_label, opt.batch_size*opt.seq_len)

    hidden = (to_var(torch.zeros(2, opt.batch_size, opt.word_embed_dim)), 
              to_var(torch.zeros(2, opt.batch_size, opt.word_embed_dim)))
    
    add_loss = 0.0 
    for t in range(num_iter_test):
        batch_input = test_generator.__next__ ()
        batch_label = label_generator.__next__()
        
        net.zero_grad()
        hidden = [state.detach() for state in hidden]
        
        test_output, hidden = net(to_var(batch_input), hidden)

        test_loss = criterion(test_output, to_var(batch_label)).data

        loss_list.append(test_loss)
        add_loss += test_loss

    print("Test Loss={0:.4f}".format(float(add_loss) / num_iter_test))
    print("Test PPL={0:.4f}".format(float(torch.exp(add_loss / num_iter_test))))


def test_Transformer(net, data, opt, model_name):
    net.eval()
 
    test_input = torch.from_numpy(data.test_input)
    test_label = torch.from_numpy(data.test_label)

    L = opt.seq_len
    num_seq = test_input.size()[0] // opt.seq_len
    test_input = test_input[:num_seq*opt.seq_len, :]
    # [num_seq, seq_len, max_word_len+2]
    test_input = test_input.view(-1, opt.seq_len, opt.max_word_len+2)

    criterion = nn.CrossEntropyLoss()

    loss_list = []
    num_iter_test = test_input.size()[0] // opt.batch_size
    test_generator = batch_generator(test_input, opt.batch_size)
    label_generator = batch_generator(test_label, opt.batch_size*opt.seq_len)
    
    X = torch.zeros((opt.batch_size,L,opt.max_word_len+2)).long().cuda()

    add_loss = 0.0 
    for t in range(num_iter_test):
        batch_input = test_generator.__next__ ()
        batch_label = label_generator.__next__()

        ####################################################################

        y_input  = batch_input[:,:,:].cuda()
        batch_label = torch.reshape(batch_label,(opt.batch_size,L)).cuda()
        y_pred      = batch_label[:,:].cuda().reshape(-1)
        ####################################################################
        
        net.zero_grad()

        test_output = net(X,y_input)

        X = y_input

        test_loss = criterion(test_output, to_var(y_pred)).data

        loss_list.append(test_loss)
        add_loss += test_loss

    print("Test Loss={0:.4f}".format(float(add_loss) / num_iter_test))
    print("Test PPL={0:.4f}".format(float(torch.exp(add_loss / num_iter_test))))

if __name__=="__main__":
    ########################## hyperparameters ##########################
    architecture = "transformer"    # LSTM | transformer
    corpus       = "Brown"   # Brown | ...

    word_embed_dim     = 525
    char_embedding_dim = 32
    USE_GPU            = True
    seq_len       = 64
    batch_size    = 16

    init_lr = 1e-4
    epochs = 20

    ########################## data preparation ##########################
    
    if corpus == "Brown":
        init_brown()

    if architecture == "LSTM":
        model_archi = charLM
    else:
        model_archi = charTransformer
    model_name = architecture+"_"+corpus+"_model"

    if not os.path.exists("results.pt"):
        results = {}
        results[model_name] = {"validation":{"loss":[],"PPL":[]},"training":{"loss":[],"PPL":[]}}
    else:
        results = torch.load("results.pt")

    if os.path.exists("cache/prep_"+corpus+".pt") is False:
        preprocess()
    objects = torch.load("cache/prep_"+corpus+".pt")
    word_dict         = objects["word_dict"]
    char_dict         = objects["char_dict"]
    reverse_word_dict = objects["reverse_word_dict"]
    max_word_len      = objects["max_word_len"]
    num_words         = len(word_dict)

    print("word/char dictionary built. Start making inputs.")

    if os.path.exists("cache/data_sets_"+corpus+".pt") is False:
        dataset_prep()
    data_sets = torch.load("cache/data_sets_"+corpus+".pt")
    train_set = data_sets["tdata"]
    valid_set = data_sets["vdata"]
    test_set  = data_sets["test"]
    train_label = data_sets["trlabel"]
    valid_label = data_sets["vlabel"]
    test_label = data_sets["tlabel"]

    DataTuple = namedtuple("DataTuple", "train_input train_label valid_input valid_label test_input test_label")
    data = DataTuple(train_input=train_set,
                    train_label=train_label,
                    valid_input=valid_set,
                    valid_label=valid_label,
                    test_input=test_set,
                    test_label=test_label)

    print("Loaded data sets. Start building network.")

    ########################## net ##########################

    net = model_archi(char_embedding_dim, 
                word_embed_dim, 
                num_words,
                len(char_dict),
                use_gpu=USE_GPU)

    for param in net.parameters():
        nn.init.uniform(param.data, -0.05, 0.05)

    Options = namedtuple("Options", [
            "init_lr", "seq_len",
            "max_word_len", "batch_size", "epochs",
            "word_embed_dim"])
    opt = Options(init_lr=init_lr,
                seq_len=seq_len,
                max_word_len=max_word_len,
                batch_size=batch_size,
                epochs=epochs,
                word_embed_dim=word_embed_dim)

    print("Network built. Start training.")

    if ("transformer" in model_name):
        train_Transformer(net, data, opt, model_name)
    else:
        train_LSTM(net, data, opt, model_name)
    torch.save(results, "results.pt")

    torch.save(net, "cache/"+model_name+"_net.pkl")
    print("saved net")

    if ("transformer" in model_name):
        test_Transformer(net, data, opt, model_name)
    else:
        test_LSTM(net, data, opt, model_name)

    model = torch.load("cache/"+model_name+"_net.pkl")
    model.eval()


