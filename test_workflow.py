import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiment_core import read_data, split_data, train_experiment, predict_file, SPECS

KIND = json.loads((Path(__file__).parent/'assignment.json').read_text())['kind']

def fixture_csv(path, count=120):
    dimensions,classes,_ = SPECS[KIND]
    rng = np.random.default_rng(7)
    x = rng.normal(size=(count,dimensions))
    labels = np.arange(count)%classes
    x[:,0] += labels*3
    df = pd.DataFrame(x,columns=[f'f{i}' for i in range(dimensions)])
    df['label'] = labels
    df.to_csv(path,index=False)
    return df

def test_split_has_no_overlap_and_scaler_is_training_only(tmp_path):
    path = tmp_path/'train.csv'; fixture_csv(path)
    x,y,_,_ = read_data(path,KIND)
    scaler,encoder,parts = split_data(x,y)
    assert sum(len(p[2]) for p in parts) == len(x)
    assert all(set(parts[i][2]).isdisjoint(parts[j][2]) for i in range(3) for j in range(i))
    np.testing.assert_allclose(scaler.mean_,x[parts[0][2]].mean(axis=0))
    assert len(encoder.classes_) == SPECS[KIND][1]
    assert len(parts[0][2]) == 72 and len(parts[1][2]) == len(parts[2][2]) == 24

def test_data_rejects_missing_nonfinite_wrong_columns_and_conflicting_duplicates(tmp_path):
    path = tmp_path/'train.csv'; df = fixture_csv(path)
    for broken in [df.drop(columns=['f0']),df.rename(columns={'f0':'wrong'}),
                   df.assign(f0=np.inf), pd.concat([df,df.iloc[[0]].assign(label=1)],ignore_index=True),
                   df.assign(label=0),df.iloc[:10]]:
        broken.to_csv(path,index=False)
        with pytest.raises(ValueError): read_data(path,KIND)
    pd.concat([df,df.iloc[[0]]]).to_csv(path,index=False)
    assert len(read_data(path,KIND)[0]) == len(df)

def test_training_reload_and_full_prediction_cli(tmp_path):
    path = tmp_path/'train.csv'; df = fixture_csv(path)
    output = tmp_path/'checkpoint'
    result = train_experiment(KIND,path,output,epochs=2)
    assert result['split_counts'] == {'train':72,'validation':24,'test':24}
    assert all(v['checkpoint_reload_identical'] for v in result['models'].values())
    predpath = tmp_path/'unlabelled.csv'; df.drop(columns=['label']).iloc[:7].to_csv(predpath,index=False)
    expected = predict_file(KIND,output,predpath,tmp_path/'direct.csv')
    run = subprocess.run([sys.executable,'experiment.py','predict','--checkpoint-dir',str(output),
                          '--predict-csv',str(predpath),'--output-csv',str(tmp_path/'cli.csv')],
                         cwd=Path(__file__).parent, capture_output=True,text=True,timeout=120)
    assert run.returncode == 0,run.stderr
    np.testing.assert_array_equal(expected,pd.read_csv(tmp_path/'cli.csv')['label'])
    with pytest.raises(ValueError): train_experiment(KIND,path,output,epochs=0)
