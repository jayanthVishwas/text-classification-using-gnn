from models.gcn import GCN
from models.gat import GAT



def get_model(args, input_dim, hidden_dim, n_classes):
    if args.model == 'GCN':
        model = GCN(input_dim, hidden_dim, n_classes)
    elif args.model == 'GAT':
        model = GAT(input_dim, hidden_dim, n_classes)

    return model
    