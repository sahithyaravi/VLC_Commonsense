import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from config import *
from networkx.drawing.nx_pydot import graphviz_layout
def construct_graph(sources, targets, relations, prune_network=False):
    df = pd.DataFrame({'source': sources, 'target': targets, 'edge': relations})
    # print(df.head(10))
    df.to_csv('srl.csv')

    G = nx.MultiGraph(directed=True)
    for i in range(len(sources)):
        G.add_edge(sources[i], targets[i], label=relations[i], len="0.2")

    if prune_network:
        # Generate connected components and select the largest:
        largest_component = max(nx.weakly_connected_components(G), key=len)

        # Create a subgraph of G consisting only of this component:
        G = G.subgraph(largest_component)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def draw_graph(G, heading="", filename='out.png'):
    # - `'dot'`
    # - `'twopi'`
    # - `'neato'`
    # - `'circo'`
    # - `'fdp'`
    plt.figure(figsize=(40, 20))
    ax = plt.gca()
    ax.set_title(heading)

    # method 1 networkx
    prog = 'twopi'
    pos = graphviz_layout(G, prog=prog)
    nx.draw(G, with_labels=True, node_color='white', node_size=5000, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'label')
    formatted_edge_labels = {(elem[0], elem[1]): edge_labels[elem] for elem in
                             edge_labels}  # use this to modify the tuple keyed dict if it has > 2 elements, else ignore
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels)
    # plt.savefig('out.png')
    plt.show()
    # method 2 pydot
    p = nx.drawing.nx_pydot.to_pydot(G)
    p.set_size('"100,100!"')
    p.write(filename, prog='dot', format='png')


def expansion_to_graphnodes(expansions_dict, sentence):
    sources = []
    targets = []
    relations = []
    print(expansions_dict)
    for relation, exp_list in expansions_dict.items():
        sources.append(sentence)
        targets.append(exp_list[0])
        relations.append(relation)
    print(targets)
    return construct_graph(sources, targets, relations)


if __name__ == '__main__':
    with open(captions_path, 'r') as fp:
        captions = json.loads(fp.read())
    with open(captions_comet_expansions_path, 'r') as fp:
        caption_expansions = json.loads(fp.read())
    with open(questions_path, 'r') as fp:
        questions = json.loads(fp.read())
    with open(questions_comet_expansions_path, 'r') as fp:
        question_expansions = json.loads(fp.read())

    df = pd.DataFrame(questions['questions'])
    df['image_id'] = df['image_id'].astype(str)
    df['question_id'] = df['question_id'].astype(str)

    chosen_id = 12
    captions_keys = list(captions.keys())
    caption_expansion_keys = list(caption_expansions.keys())
    image_name = captions_keys[chosen_id]
    image_id = image_path_to_id(image_name)
    df_img = df.loc[df['image_id'] == image_id]
    for idx, row in df_img.iterrows():
        question = row['question']
        qn_exp = question_expansions[row['question_id']]
        qn_graph = expansion_to_graphnodes(qn_exp, question)
        draw_graph(qn_graph)
        break

    graph = expansion_to_graphnodes(caption_expansions[captions_keys[100]], captions[captions_keys[100]])
    draw_graph(graph, filename='caption.png')