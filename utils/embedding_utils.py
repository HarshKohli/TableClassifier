# Author: Harsh Kohli
# Date Created: 06-05-2021

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


def index_embeddings(id2_embed, train_index):
    dim = id2_embed[0].size
    t = AnnoyIndex(dim, 'angular')
    for id, embedding in id2_embed.items():
        t.add_item(id, embedding)
    t.build(30)
    t.save(train_index)


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
                hardest_negative = {'table_id': cur_id, 'question_tokens': close_sample['question_tokens'], 'label': 0}
                hardest_negatives.append(hardest_negative)
                break
    return hardest_negatives
