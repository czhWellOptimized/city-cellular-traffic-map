from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import logging

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape # (34272, 207)
    print(num_samples) # 192 
    print(num_nodes) # 2617
    data = np.expand_dims(df.values, axis=-1) # (192, 2617, 1)
    print(data.shape)
    data_list = [data]
    print(df.index.values.shape) # 192
    # print(df.index.values)
    print(type(df.index.values[0])) # <class 'numpy.datetime64'>
# ['2012-08-19T00:00:00.000000000' '2012-08-19T01:00:00.000000000'
#  '2012-08-19T02:00:00.000000000' '2012-08-19T03:00:00.000000000'
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D") #0.04166667
        # print(time_ind)
        # logging.warning(time_ind.shape) # 34272
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) 
        # logging.warning(time_in_day.shape) # (34272, 207, 1)
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1) # (192, 2617, 2)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    # [-(T-1),0] # [1,T]
    min_t = abs(min(x_offsets)) # T-1
    max_t = abs(num_samples - abs(max(y_offsets)))  # 192 - T
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...] #  [0,T-1] ... [192-2*T...191-T]
        y_t = data[t + y_offsets, ...] #  [T,2*T-1]... [192-T...191]
        x.append(x_t)
        y.append(y_t)
    # print(len(x)) # list len = 193-2*T = 
    x = np.stack(x, axis=0) # (193-2*T, T, 2617, 2) B T N F
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename) # 对于h5文件应该采用该方法直接简单
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-7, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 9, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    # print(x_offsets.shape)
    # print(x_offsets.reshape(list(x_offsets.shape) + [1]))
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]), # [-11,0]
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]), # [1,12]
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)