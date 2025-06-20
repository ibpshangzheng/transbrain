"""Graph embedding generator using Node2Vec algorithm."""
import os
import argparse
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from pipeline.code.graph_walk.connectomeembeddings.connectome_embed_nature import create_embedding

def parse_arguments() -> dict:
    """
    Parse command line parameters for embedding generation.

    Returns
    -------
    dict
        Dictionary containing validated input parameters.
    """
    parser = argparse.ArgumentParser(
        description='Generate graph embeddings from connectivity matrix',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_argument_group('Input/Output')
    input_group.add_argument('-i', '--input', required=True,
                           help='Path to connectivity matrix CSV')
    input_group.add_argument('-s', '--save_dir', required=True,
                           help='Output directory path')

    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('-d', '--dimensions', type=int, default=40,
                           help='Embedding vector dimensions')
    model_group.add_argument('-wl', '--walk_length', type=int, default=40,
                           help='Random walk sequence length')
    model_group.add_argument('-ws', '--window_size', type=int, default=5,
                           help='Context window size for training')
    model_group.add_argument('-p', '--return_param', type=float, default=0.01,
                           help='Return hyperparameter (p)')
    model_group.add_argument('-q', '--inout_param', type=float, default=0.1,
                           help='In-out hyperparameter (q)')
    model_group.add_argument('-pm', '--permutations', type=int, default=15,
                           help='Number of random walk permutations')

    args = parser.parse_args()
    
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input matrix not found: {args.input}")
        
    return vars(args)

def generate_embeddings(params: dict) -> None:
    """
    Generate graph embeddings.

    Parameters
    ----------
    params : dict
        Dictionary containing parameters for embedding generation, including:
        
        input : str
            Path to connectivity matrix CSV.
        save_dir : str
            Output directory path.
        dimensions : int
            Embedding vector dimensions.
        walk_length : int
            Random walk sequence length.
        window_size : int
            Context window size for training.
        return_param : float
            Return hyperparameter (p).
        inout_param : float
            In-out hyperparameter (q).
        permutations : int
            Number of random walk permutations.

    Returns
    -------
    None
    """
    
    base_dir = Path(params['save_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    info = f"Human_Mouse_p{params['return_param']}_q{params['inout_param']}"
    embed_path = base_dir / info

    ce_matrix = pd.read_csv(params['input'], index_col=0).values
    graph = nx.DiGraph(ce_matrix)
    print("Graph connectivity:", nx.is_weakly_connected(graph))

    embedding_ = create_embedding(
        str(base_dir),                   
        str(embed_path.with_name(f"{info}_graph_only_positive.txt")), 
        str(embed_path.with_name(f"{info}_graph")), 
        ce_matrix,
        f"{info}_graph_embeddings",  
        permutation_no=params['permutations'],
        dimensions=params['dimensions'],
        walk_length=params['walk_length'],
        window_size=params['window_size'],
        p=params['return_param'],
        q=params['inout_param']
    )
    
    with open(embed_path.with_name(f"{info}_graph_embeddings.pkl"), "wb") as f:
        pickle.dump(embedding_, f)

if __name__ == '__main__':
    config = parse_arguments()
    generate_embeddings(config)