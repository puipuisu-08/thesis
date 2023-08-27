import pandas as pd
import os
import gc

def splitter(args):
    source_raw = pd.read_csv(os.path.join('./dataset/', args.source_path))
    target_raw = pd.read_csv(os.path.join('./dataset/', args.target_path))

    size = target_raw.shape[0]
    for i in range(args.num_sources):
        source_raw.iloc[i * size:(i + 1) * size].to_csv(os.path.join('./dataset/', f'source_{i}.csv'), index=False)
    source_raw.to_csv(os.path.join('./dataset/', 'source.csv'), index=False)
    target_raw.to_csv(os.path.join('./dataset/', 'target.csv'), index=False)

    del source_raw, target_raw
    gc.collect()