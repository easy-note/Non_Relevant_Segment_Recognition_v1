import os
import pandas as pd


def save_OOB_result_csv(metric, epoch, args, save_path):
    m_keys = list(metric.keys())

    cols = ['Model', 'Epoch', *m_keys]

    save_path = save_path + '/result.csv'
    model_name = args.model
    
    data = [model_name, epoch, *list(metric[key] for key in m_keys)]

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        print('Existed file loaded')
        
        new_df = pd.Series(data, index=cols)
        df = df.append(new_df, ignore_index=True)
        print('New line added')
        
    else:
        print('New file generated!')
        df = pd.DataFrame([data],
                    columns=cols
                    ) 

    df.to_csv(save_path, 
            index=False,
            float_format='%.4f')
