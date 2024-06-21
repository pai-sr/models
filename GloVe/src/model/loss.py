import torch

def glove_loss(focal_embed, context_embed, focal_bias, context_bias, counts, x_max, alpha):
    weight_factor = torch.pow(counts / x_max, alpha)
    weight_factor[weight_factor > 1] = 1

    embedding_products = torch.sum(focal_embed * context_embed, dim=1)
    log_cooccurrences = torch.log(counts)

    distance_expr = (embedding_products + focal_bias + context_bias + log_cooccurrences) ** 2
    loss = weight_factor * distance_expr
    return torch.mean(loss)