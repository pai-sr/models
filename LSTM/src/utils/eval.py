
def eval_epoch(value_name, value, data_len, batch_size, output_dict=True):
    if output_dict:
        return {value_name : value / (data_len / batch_size)}
    else:
        return value / (data_len / batch_size)