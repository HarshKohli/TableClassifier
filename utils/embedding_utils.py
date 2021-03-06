# Author: Harsh Kohli
# Date Created: 06-05-2021

import os
from annoy import AnnoyIndex


def compute_embeddings(text, nnlm_embedder, batch_size):
    batches = [text[i * batch_size:(i + 1) * batch_size] for i in
               range((len(text) + batch_size - 1) // batch_size)]
    num_batches = len(batches)
    print('Computing Embeddings...')
    embeddings = []
    for index, batch in enumerate(batches):
        if index % 100 == 0:
            print('Done ' + str(index) + ' batches out of ' + str(num_batches))
        embedding = nnlm_embedder(batch)
        embeddings.extend(embedding)
    return embeddings


def index_embeddings(id2_embed, index_file, dim):
    t = AnnoyIndex(dim, 'angular')
    for id, embedding in id2_embed.items():
        t.add_item(id, embedding)
    t.build(30)
    dir_name = os.path.dirname(index_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    t.save(index_file)


def get_hardest_negatives(samples_data, train_index, dim):
    u = AnnoyIndex(dim, 'angular')
    u.load(train_index)
    hardest_negatives = []
    for index, sample in enumerate(samples_data):
        similar_questions = u.get_nns_by_item(index, 1000)
        cur_id = sample['table_id']
        for close_index in similar_questions:
            close_sample = samples_data[close_index]
            close_id = close_sample['table_id']
            if cur_id != close_id:
                hardest_negative = {'table_id': cur_id, 'question_tokens': close_sample['question_tokens'],
                                    'label': 0.0}
                hardest_negatives.append(hardest_negative)
                break
    return hardest_negatives


def get_query_table_ranks(sample_info_dict, id_to_index, index_file, dim):
    u = AnnoyIndex(dim, 'angular')
    u.load(index_file)
    ranks = []
    for sentence, info in sample_info_dict.items():
        table_id, embedding = info['table_id'], info['embedding']
        table_index = id_to_index[table_id]
        closest_tables = u.get_nns_by_vector(embedding, 1000000)
        rank = closest_tables.index(table_index) + 1
        ranks.append(rank)
    return ranks


def get_top_k_tables(sample_info_dict, id_to_index, index_file, dim, k):
    u = AnnoyIndex(dim, 'angular')
    u.load(index_file)
    ranks, top_k = [], {}
    for sentence, info in sample_info_dict.items():
        table_id, embedding = info['table_id'], info['embedding']
        table_index = id_to_index[table_id]
        closest_tables = u.get_nns_by_vector(embedding, 1000000)
        rank = closest_tables.index(table_index)
        if rank < k:
            label = [0 for _ in range(k)]
            label[rank] = 1
            info['top_k'] = closest_tables[:k]
            info['labels'] = label
        ranks.append(rank)
    return ranks
