import pickle
import torch
import numpy as np
import collections
import argparse 
import torch.optim as optim 
from utils import get_topwords, plot_fig
from plsv_vae import PlsvVAE
from sklearn.feature_extraction.text import CountVectorizer


def test(model, tensor_train_w, train_label, all_indices, categories):
    model.eval()
    x_list = []
    phi_list = []
    beta_list = []
    labels_list = []
    with torch.no_grad():
        for batch_ndx in all_indices:
            input_w = tensor_train_w[batch_ndx].to(device)
            labels = train_label[batch_ndx]
            labels_list.extend(labels)
            z, recon_v, zx, zx_phi = model(input_w, compute_loss=False)
            zx = zx.view(-1, num_coordinate).data.detach().cpu().numpy()
            x_list.extend(zx)

        x_list = np.array(x_list)

        beta = model.decoder.weight.data.cpu().numpy().T#
        zphi = model.decoder_phi_bn(model.centres).data.cpu().numpy()
        print("---"*10)
        topword_topics = get_topwords(beta, id_vocab)
        for topwords in topword_topics:
            print(topwords)
        print("---"*10)

    return x_list, zphi, labels_list, topwords_topic
    

def load_reuters():
    with open('data/reuters/preprossed_data.pkl', 'rb') as f: 
        preprossed_data = pickle.load(f)

    with open('data/reuters/data_reuters_labels.pkl', 'rb') as f:
        train_label = pickle.load(f)

    vectorizer = CountVectorizer(min_df=9)
    train_vec = vectorizer.fit_transform(preprossed_data).toarray()
    vocab = vectorizer.vocabulary_
    nonzeros_indexes = np.where(train_vec.any(1))[0]
    train_vec_non_zeros = [train_vec[i] for i in nonzeros_indexes]

    train_vec = np.array(train_vec_non_zeros)
    return train_vec, train_label, vocab
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-iters',        type=int,   default=1)
parser.add_argument('-t', '--topics',        type=int,   default=25)
parser.add_argument('-bs', '--batch-size',        type=int,   default=256)
parser.add_argument('-h1', '--hidden1',        type=int,   default=100)
parser.add_argument('-h2', '--hidden2',        type=int,   default=100)
parser.add_argument('-e', '--epochs',        type=int,   default=100)
parser.add_argument('-d', '--n-coordinates',        type=int,   default=2)
parser.add_argument('-var', '--varx',        type=int,   default=1)
parser.add_argument('-lr', '--learning-rate',        type=float,   default=0.002)
parser.add_argument('-dataset', '--dataset',        type=str,   default="arxiv")
parser.add_argument('-dist', '--distance',        type=str,   default="gaussian")
parser.add_argument('-name', '--machine-name',        type=str,   default="titan")

args = parser.parse_args()

if __name__ == '__main__':
    
    train_vec, train_label, vocab = load_reuters()
    tensor_train_w = torch.from_numpy(np.array(train_vec)).float()
    train_label = np.asarray(train_label)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    id_vocab = dict(map(reversed, vocab.items()))
    
    beta1 = 0.99
    beta2 = 0.999
    drop_rate = 0.6
    en1_units_x = args.hidden1
    en2_units_x = args.hidden2
    num_coordinate = args.n_coordinates
    num_input = train_vec[0].shape[0]
    variance_x = args.varx
    learning_rate = args.learning_rate
    num_topic = args.topics
    bs = args.batch_size
    epochs = args.epochs
    distance = args.distance

    model = PlsvVAE(num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, drop_rate, variance_x, bs, distance)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, betas=(beta1, beta2))
    phi_arr = []
    beta_arr = []
    x_arr = []
    recon_arr = []
    list_d = []
    all_indices = torch.randperm(tensor_train_w.size(0)).split(bs)

    for epoch in range(epochs):

        loss_u_epoch = 0.0
        loss_xkl_epoch = 0.0
        loss_betakl_epoch = 0.0
        loss_phikl_epoch = 0.0
        loss_KLD = 0.0
        loss_phi_epoch = 0.0
        recon_ep = 0.0
        loss_epoch = 0.0
        model.train()
        d_temp = []
        for batch_ndx in all_indices:

            input_w = tensor_train_w[batch_ndx].to(device)
            labels = train_label[batch_ndx]
            input_c = None
            recon_v, (loss, loss_u, xkl_loss, kl) = model(input_w, compute_loss=True)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()             # backpror.step()            # update parameters
            loss_epoch += loss.item()
            loss_u_epoch += loss_u.item()
            loss_xkl_epoch += xkl_loss.item() 
            loss_KLD += kl.item() #/ len(tensor_train_w)) 
            recon_ep += recon_v.mean().item() 

        x_arr.append(loss_xkl_epoch)
        recon_arr.append(loss_u_epoch)

        if epoch % 10 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch))
            print('x_loss={}, recon_loss={}, KLD= {}'.format(loss_xkl_epoch, loss_u_epoch, loss_KLD)) 
