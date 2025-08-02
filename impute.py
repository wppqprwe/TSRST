import argparse
import multiprocessing

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np
from einops import reduce
import scanpy as sc
import pickle
import pandas as pd
import warnings
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
from visual import plot_spot_masked_image, plot_clusters
from utils import (
        load_tsv, load_pickle, save_pickle, load_image, save_image,
        read_lines, setToArray, build_her2st_data)


class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            # TODO: change activation to LeakyRelu(0.01)
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class ForwardSumModel(pl.LightningModule):

    def __init__(self, lr, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.lr = lr
        self.x_embed = nn.Embedding(2000, 2)
        self.y_embed = nn.Embedding(2000, 2)

        self.to_latent = nn.Sequential(
                FeedForward(input_dim, hidden_dim),
        )

        self.net_lat = nn.Sequential(
                FeedForward(hidden_dim+4, hidden_dim+4),
                FeedForward(hidden_dim+4, hidden_dim+4),
                FeedForward(hidden_dim+4, hidden_dim+4))

        self.net_out = FeedForward(
                hidden_dim+4, out_dim,
                activation=ELU(alpha=0.01, beta=0.01))
        self.save_hyperparameters()

    def forward(self, x, loc, Train=True):
        loc_x = loc[..., 0].long()
        loc_y = loc[..., 1].long()
        if Train:
            x_embedding = self.x_embed(loc_x).unsqueeze(1).expand(-1, x.size(1), -1)
            y_embedding = self.y_embed(loc_y).unsqueeze(1).expand(-1, x.size(1), -1)
        else:
            x_embedding = self.x_embed(loc_x)
            y_embedding = self.y_embed(loc_y)

        x = self.to_latent(x)
        x = torch.cat((x, x_embedding, y_embedding), dim=-1)
        x = self.net_lat(x)
        x = self.net_out(x)
        return x

    def training_step(self, batch, batch_idx):
        # x:(27,121,579) y_mean:(27,1000)
        x, y_mean, loc = batch
        y_pred = self.forward(x, loc)    # y_pred:(27,121,1000)
        y_mean_pred = y_pred.mean(-2)
        # TODO: try l1 loss
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('mse', mse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius, indices=None):
        super().__init__()
        mask = get_disk_mask(radius)    # mask:(15,15) radius=6.25上取整7 为spot的半径
        x = get_patches_flat(x_all, locs, mask) # (441,121,579) 代表每个spot的图像特征数据
        isin = np.isfinite(x).all((-1, -2))
        self.x = x[isin]
        self.y = y[isin]
        self.locs = locs[isin]
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask
        if indices:
            self.x = x[indices]
            self.y = y[indices]
            self.locs = locs[indices]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.locs[idx] # x[idx]:(121, 579) y[idx]:(1000,)

    def show(self, channel_x, channel_y, prefix):
        mask = self.mask
        size = self.size
        locs = self.locs
        xs = self.x
        ys = self.y

        plot_spot_masked_image(
                locs=locs, values=xs[:, :, channel_x], mask=mask, size=size,
                outfile=f'{prefix}x{channel_x:04d}.png')

        plot_spot_masked_image(
                locs=locs, values=ys[:, channel_y], mask=mask, size=size,
                outfile=f'{prefix}y{channel_y:04d}.png')



def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]  # patch:(15,15,579)
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda'):

    print('x:', x.shape, ', y:', y.shape)   # x: (528, 704, 579) , y: (441, 1000)

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius)

    # train_indices, test_indices = train_test_split(range(len(y)), test_size=0.2)
    # dataset = SpotDataset(x, y, locs, radius, train_indices)
    # test_dataset = SpotDataset(x, y, locs, radius, test_indices)

    dataset.show(
            channel_x=0, channel_y=0,
            prefix=f'{prefix}training-data-plots/')

    model = train_load_model(
            model_class=ForwardSumModel,
            model_kwargs=dict(
                input_dim=x.shape[-1],
                hidden_dim=256,
                out_dim=y.shape[-1],
                lr=lr),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset


def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)


def predict(
        model_states, image, x_batches, name_list, prefix,
        device='cuda'):

    # states: different initial values for training
    # batches: subsets of observations
    # groups: subsets outcomes

    model_states = [mod.to(device) for mod in model_states]
    H, W, C = image.shape
    patch_size = 11

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    locs = np.stack([y_coords, x_coords], axis=-1)

    batch_size_row = 50
    n_batches_row = locs.shape[0] // batch_size_row + 1
    locs_batches = np.array_split(locs, n_batches_row)

    result = []
    with torch.no_grad():
        for i, mod in enumerate(model_states):
            recon_img = []
            for patch, loc in zip(x_batches, locs_batches):
                patch = torch.tensor(patch, device=device).float() 
                loc = torch.tensor(loc, device=device).float()
                pred = mod(patch, loc, Train=False)
                recon_img.append(pred.cpu())
            # recon_img_cpu = [tensor.cpu() for tensor in recon_img]
            recon_img = torch.cat(recon_img, dim=0)
            result.append(recon_img)
    result = [a.cpu().numpy() for a in result]
    # y_grp = np.median(result,0)
    y_grp = result[0]
    print('saving embedding-gene')
    save_pickle(y_grp,prefix+'embeddings-gene.pickle')


    for i, name in tqdm(enumerate(name_list), total=len(name_list)):
        save_pickle(y_grp[..., i], f'{prefix}cnts-super/{name}.pickle')

    # for i, name in tqdm(enumerate(name_list), total=len(name_list)):
    #     save_pickle(np.median([a[..., i] for a in result],0), f'{prefix}cnts-super/{name}.pickle')
    


def impute(
        embs, cnts, locs, radius, epochs, batch_size, prefix, args,
        n_states=1, load_saved=False, device='cuda', n_jobs=1):

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    # __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    kwargs_list = [
            dict(
                x=embs, y=cnts, locs=locs, radius=radius,
                batch_size=batch_size, epochs=epochs, lr=1e-4,
                prefix=f'{prefix}states/{i:02d}/',
                load_saved=load_saved, device=device)
            for i in range(n_states)]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    # model输入为x(441,121,579)表示441个spot的patch图像的特征,输出为y(441,1000),表示441个spot的1000个基因的表达值
    model_list = [out[0].to(device) for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    # cnts_range = np.stack([cnts_min, cnts_max], -1)

    batch_size_row = 50
    n_batches_row = embs.shape[0] // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)

    predict(
            model_states=model_list,image=embs, x_batches=embs_batches,
            name_list=names, #y_range=cnts_range,
            prefix=prefix, device=device)

    if args.testing:
        print("Testing!")
        mse_gene = []
        mae_gene = []
        pcc_gene = []
        mod, dataset = model_list[0], dataset_list[0]
        with torch.no_grad():
            x = torch.tensor(dataset.x, device=device).float() 
            loc = torch.tensor(dataset.locs, device=device).float()
            y = torch.tensor(dataset.y, device=device).float()
            pred = mod(x, loc).mean(-2)
        # pred *= torch.tensor(cnts_range[:, 1] - cnts_range[:, 0], device=device)
        # pred += torch.tensor(cnts_range[:, 0], device=device)
        for i in range(y.size(1)):
            # non_zero_mask = torch.ne(y[:,i], 0)
            # y_gene = y[non_zero_mask,i]
            # pred_gene = pred[non_zero_mask,i]
            y_gene = y[:,i]
            pred_gene = pred[:,i]
            mse = ((pred_gene - y_gene)**2).mean().cpu().detach().numpy()
            mae = (abs(pred_gene - y_gene)).mean().cpu().detach().numpy()
            pcc, _ = pearsonr(pred_gene.flatten().cpu().detach().numpy(), y_gene.flatten().cpu().detach().numpy())
            mse_gene.append(np.round(mse,4))
            mae_gene.append(np.round(mae,4))
            pcc_gene.append(np.round(pcc,4))
        mse_gene = np.array(mse_gene)
        mae_gene = np.array(mae_gene)
        pcc_gene = np.array(pcc_gene)
        np.save(f'{prefix}mse_gene.npy', mse_gene)
        np.save(f'{prefix}mae_gene.npy', mae_gene)
        np.save(f'{prefix}pcc_gene.npy', pcc_gene)

        

    if args.cluster:
        print("Clustering!")

        with open(f'{prefix}truth.txt', 'r') as file:
            ground_truth = []
            for line in file:
                numbers = line.split()
                numbers = [int(num)-1 if num != 'NA' else 7 for num in numbers]
                ground_truth.extend(numbers)
        factor = args.factor # 对原图片缩放的倍数:(8448,11264)->(528,704)
        radius = args.radius
        radius = np.round(radius / factor).astype(int)
        
        img = load_image(f'{prefix}image.jpg')
        img = reduce(
                img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
                h=args.factor, w=args.factor).astype(np.uint8)

        plot_clusters(img=img, locs=locs, radius=radius, 
                        outfile=f'{prefix}ground_truth.png', clusters=ground_truth)
        
        with torch.no_grad():
            x = torch.tensor(dataset_list[0].x, device=device).float()
            loc = torch.tensor(dataset_list[0].locs, device=device).float()
            y = torch.tensor(dataset_list[0].y, device=device).float()
            pred = model_list[0](x, loc).mean(-2)

        latent = pred.cpu().detach().numpy()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(latent)
        pca = PCA(n_components=100)
        latent = pca.fit_transform(data_scaled)

        n_clusters = 7
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(latent)
        labels = kmeans.labels_
        plot_clusters(img=img, locs=locs, radius=radius, 
                        outfile=f'{prefix}clusters.png', clusters=labels)
        ari = adjusted_rand_score(ground_truth, labels)
        print(f'ARI: {ari}')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)  # e.g. 400
    parser.add_argument('--n-states', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--load-saved', action='store_true')
    parser.add_argument('--testing', type=int, default=0)
    parser.add_argument('--cluster', type=int, default=0)
    parser.add_argument('--factor', type=int, default=16)
    parser.add_argument('--radius', type=int, default=120)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    embs = load_pickle(f'{args.prefix}embeddings-hist.pickle')
    # embs = np.concatenate([embs['emb']])
    embs = np.concatenate([embs['cls'], embs['sub'], embs['rgb']])
    embs = embs.transpose(1, 2, 0)
    embs = np.array(embs)

    # visium data
    adata = sc.read_visium(path=args.prefix,count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()

    # her2st data
    # adata = build_her2st_data(args.prefix, 'H3')

    # normalize
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cnts = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    # cnts = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    # gene_names = read_lines(f'{args.prefix}tls_marker.txt')
    gene_names = list(adata.var[adata.var['highly_variable']].index)
    cnts = cnts[gene_names]
    
    locs = adata.obsm['spatial']
    locs = np.stack([locs[:,1], locs[:,0]], -1)
    locs *= 2

    # sample_index = np.random.choice(range(cnts.shape[0]), size=round(0.5 * cnts.shape[0]),replace=False)
    # sample_index = setToArray(set(sample_index))
    # cnts = cnts.iloc[sample_index]
    # locs = locs[sample_index]

    factor = args.factor # 对原图片缩放的倍数:(8448,11264)->(528,704)
    # radius = int(read_string(f'{args.prefix}radius.txt'))
    radius = args.radius
    radius = radius / factor
    locs = (locs / factor).round().astype(int)

    n_train = cnts.shape[0]
    batch_size = min(128, n_train//16)  # 27


    impute(
            embs=embs, cnts=cnts, locs=locs, radius=radius, args=args,
            epochs=args.epochs, batch_size=batch_size,
            n_states=args.n_states, prefix=args.prefix,
            load_saved=args.load_saved,
            device=args.device, n_jobs=args.n_jobs)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
