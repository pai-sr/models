
def eval_epoch(value_name, value, data_len, output_dict=True):
    if output_dict:
        return {value_name : value / (data_len)}
    else:
        return value / (data_len)