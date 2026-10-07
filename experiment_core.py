"""Portable train/validation/test workflow; raw project data stays local."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import models

SPECS = {'numpy': (23, 4, 200), 'optimizer': (25, 4, 80),
         'regularization': (27, 5, 60), 'convolution': (29, 6, 50), 'recurrent': (35, 4, 45)}

def read_data(path, kind, labelled=True):
    p = Path(path)
    if not p.is_file() or p.stat().st_size > 100_000_000:
        raise ValueError('Expected a local CSV smaller than 100 MB')
    df = pd.read_csv(p)
    count, classes, _ = SPECS[kind]
    columns = [f'f{i}' for i in range(count)]
    required = columns + (['label'] if labelled else [])
    if list(df.columns) != required or df.empty or df.isna().any().any():
        raise ValueError('CSV needs ordered feature columns and, for training, a nonmissing label column')
    x = df[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Non-finite features')
    labels = None
    if labelled:
        # Do not allow identical feature vectors into more than one split.
        if df.groupby(columns, dropna=False)['label'].nunique().max() > 1:
            raise ValueError('Identical features have conflicting labels')
        df = df.drop_duplicates(subset=columns)
        x = df[columns].to_numpy(dtype=np.float64)
        labels = df['label'].to_numpy()
        if len(np.unique(labels)) != classes: raise ValueError('Unexpected number of classes')
        if df['label'].value_counts().min() < 10: raise ValueError('Need at least ten distinct samples per class')
    return x, labels, columns, hashlib.sha256(p.read_bytes()).hexdigest()

def split_data(x, labels, seed=42):
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    indices = np.arange(len(x))
    train_val, test = train_test_split(indices, test_size=.2, stratify=y, random_state=seed)
    train, val = train_test_split(train_val, test_size=.25, stratify=y[train_val], random_state=seed)
    scaler = StandardScaler().fit(x[train])
    return scaler, encoder, [(scaler.transform(x[i]), y[i], i) for i in (train, val, test)]

def score(y, probs):
    if probs.ndim != 2 or len(y) != len(probs) or not np.isfinite(probs).all():
        raise ValueError('Invalid predictions')
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5): raise ValueError('Probabilities do not sum to one')
    pred = probs.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'cross_entropy': float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean()),
            'confusion_matrix': confusion_matrix(y, pred, labels=np.arange(probs.shape[1])).tolist()}

def numpy_fit(x, y, vx, vy, epochs):
    np.random.seed(42)
    params = models.init_params([23, 64, 32, 4])
    best, best_loss, best_epoch, history = None, math.inf, 0, []
    for epoch in range(1, epochs+1):
        order = np.random.permutation(len(x))
        for start in range(0, len(x), 64):
            bx, by = x[order[start:start+64]], y[order[start:start+64]]
            probs, cache = models.forward(bx, params, training=True)
            grads = models.backward(probs, by, params, cache)
            for l in range(1, len(params)//2+1): grads[f'dW{l}'] += .02*params[f'W{l}']
            models.update_params(params, grads, .01*(.98**(epoch//50)))
        probs, _ = models.forward(vx, params, training=False)
        loss = score(vy, probs)['cross_entropy']
        if not all(np.isfinite(v).all() for v in params.values()): raise ValueError('Training diverged')
        history.append({'epoch': epoch, 'validation_cross_entropy': loss})
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best = {k: v.copy() for k, v in params.items()}
    return best, best_epoch, history

def torch_factories(kind):
    import torch
    if kind == 'optimizer':
        return {name: (models.MLP, lambda p, name=name: getattr(models, name)(p, lr=.01 if name in ('SGD','SGDMomentum') else .001))
                for name in ('SGD', 'SGDMomentum', 'RMSProp', 'Adam')}
    if kind == 'regularization':
        return {name: (getattr(models, name), lambda p: torch.optim.Adam(p, lr=.001))
                for name in ('BaselineMLP', 'DropoutMLP', 'BNormMLP', 'CombinedMLP')}
    factory = models.MyCNN if kind == 'convolution' else models.LSTMClassifier
    return {factory.__name__: (factory, lambda p: torch.optim.Adam(p, lr=.001))}

def as_tensor(x, kind):
    import torch
    t = torch.as_tensor(x, dtype=torch.float32)
    if kind == 'convolution': t = torch.nn.functional.pad(t, (0, 49-t.shape[1])).reshape(-1,1,7,7)
    elif kind == 'recurrent': t = t.unsqueeze(-1)
    return t

def torch_probs(model, x):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).softmax(dim=1) for batch in x.split(256)]).numpy()

def torch_fit(kind, factory, optimizer_factory, x, y, vx, vy, epochs):
    import torch
    torch.manual_seed(42)
    model = factory()
    optimizer = optimizer_factory(model.parameters())
    tx, tv, ty = as_tensor(x,kind), as_tensor(vx,kind), torch.tensor(y, dtype=torch.long)
    best, loss_best, epoch_best, history = None, math.inf, 0, []
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(tx))
        batches = list(order.split(64))
        # BatchNorm cannot estimate variance from one row. Merge the tail rather than drop a sample.
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = torch.cat(batches[-2:]); batches.pop()
        for indices in batches:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(tx[indices]), ty[indices])
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        value = score(vy, torch_probs(model, tv))['cross_entropy']
        history.append({'epoch':epoch, 'validation_cross_entropy':value})
        if value < loss_best:
            loss_best, epoch_best = value, epoch
            best = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-epoch_best >= 12: break
    model.load_state_dict(best)
    return model, epoch_best, history

def train_experiment(kind, path, output, epochs=None):
    epochs = SPECS[kind][2] if epochs is None else epochs
    if not isinstance(epochs,int) or not 1 <= epochs <= 1000: raise ValueError('Epoch count must be 1..1000')
    x, labels, columns, source_sha = read_data(path, kind)
    scaler, encoder, parts = split_data(x, labels)
    (tx,ty,ti), (vx,vy,vi), (xx,yy,xi) = parts
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    meta = {'kind':kind,'columns':columns,'classes':encoder.classes_.tolist(),
            'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()}
    results = {'seed':42,'source_sha256':source_sha,'distinct_samples':len(x),
               'split_counts':{'train':len(tx),'validation':len(vx),'test':len(xx)},
               'selection':'lowest validation cross-entropy, never test accuracy',
               'official_unlabelled_test_accuracy':'unavailable','models':{}}
    if kind == 'numpy':
        params, epoch, history = numpy_fit(tx,ty,vx,vy,epochs)
        vp = models.forward(vx,params,training=False)[0]
        test_probs = models.forward(xx,params,training=False)[0]
        np.savez(output/'NumPyMLP.npz', **params)
        with np.load(output/'NumPyMLP.npz', allow_pickle=False) as archive:
            restored = {k:archive[k] for k in archive.files}
        np.testing.assert_array_equal(test_probs,models.forward(xx,restored,training=False)[0])
        results['models']['NumPyMLP'] = {'selected_epoch':epoch,'validation':score(vy,vp),'test':score(yy,test_probs),'history':history,'checkpoint_reload_identical':True}
    else:
        import torch
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        for name,(factory,optimizer) in torch_factories(kind).items():
            model, epoch, history = torch_fit(kind,factory,optimizer,tx,ty,vx,vy,epochs)
            vp, test_probs = torch_probs(model,as_tensor(vx,kind)), torch_probs(model,as_tensor(xx,kind))
            torch.save(model.state_dict(),output/(name+'.pt'))
            restored = factory()
            restored.load_state_dict(torch.load(output/(name+'.pt'),map_location='cpu',weights_only=True))
            np.testing.assert_array_equal(test_probs,torch_probs(restored,as_tensor(xx,kind)))
            results['models'][name] = {'selected_epoch':epoch,'validation':score(vy,vp),'test':score(yy,test_probs),'history':history,'checkpoint_reload_identical':True}
            print(f'{name}: selected epoch {epoch}, held-out accuracy {results["models"][name]["test"]["accuracy"]:.4f}',flush=True)
    meta['selected_model'] = min(results['models'],key=lambda n:results['models'][n]['validation']['cross_entropy'])
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results

def predict_file(kind, checkpoint_dir, csv, destination):
    folder = Path(checkpoint_dir)
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if meta['kind'] != kind: raise ValueError('Checkpoint project mismatch')
    x,_,cols,_ = read_data(csv,kind,labelled=False)
    if cols != meta['columns']: raise ValueError('Feature order mismatch')
    x = (x-np.array(meta['mean']))/np.array(meta['scale'])
    name = meta['selected_model']
    if kind == 'numpy':
        if name != 'NumPyMLP': raise ValueError('Unknown model')
        with np.load(folder/(name+'.npz'),allow_pickle=False) as archive:
            params = {k:archive[k] for k in archive.files}
        probs = models.forward(x,params,training=False)[0]
    else:
        import torch
        torch.set_num_threads(2)
        if name not in torch_factories(kind): raise ValueError('Unknown model')
        model = torch_factories(kind)[name][0]()
        model.load_state_dict(torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True))
        probs = torch_probs(model,as_tensor(x,kind))
    if not np.isfinite(probs).all(): raise ValueError('Non-finite predictions')
    pred = np.array(meta['classes'])[probs.argmax(axis=1)]
    pd.DataFrame({'label':pred}).to_csv(destination,index=False)
    return pred

def main(kind):
    p = argparse.ArgumentParser(description='Completed project experiment; private datasets and checkpoints stay local.')
    sub = p.add_subparsers(dest='command',required=True)
    tr = sub.add_parser('train'); tr.add_argument('--train-csv',required=True); tr.add_argument('--output-dir',required=True); tr.add_argument('--epochs',type=int)
    pr = sub.add_parser('predict'); pr.add_argument('--checkpoint-dir',required=True); pr.add_argument('--predict-csv',required=True); pr.add_argument('--output-csv',required=True)
    args = p.parse_args()
    try:
        if args.command == 'train': train_experiment(kind,args.train_csv,args.output_dir,args.epochs)
        else: predict_file(kind,args.checkpoint_dir,args.predict_csv,args.output_csv)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, 'Experiment failed: '+str(exc)+'\n')
