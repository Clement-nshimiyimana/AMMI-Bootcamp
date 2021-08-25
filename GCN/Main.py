from Train import train_node_classifier
from Utils import *

def print_results(result_dict):
    if "train" in result_dict:
        print("Loss: ",(result_dict['err']).item())
        print("Train accuracy: %4.2f%%" % (100.0*result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0*result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0*result_dict["test"]))


if __name__ == "__main__":
    node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
                                                        dataset=dataset,
                                                        c_hidden=16,
                                                        num_layers=2,
                                                        dp_rate=0.5)

    print_results(node_mlp_result)

    node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN",
                                                        layer_name="GCN",
                                                        dataset=dataset, 
                                                        c_hidden=16, 
                                                        num_layers=2,
                                                        dp_rate=0.5)
    print_results(node_gnn_result)