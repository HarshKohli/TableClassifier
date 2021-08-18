# Author: Harsh Kohli
# Date Created: 13-05-2021

import numpy as np
from utils.embedding_utils import get_query_table_ranks


def train_epoch(questions, headers, table_words, labels, all_num_cols, masks, train_iterations, train_step):
    for index, (question, header, table_word, label, all_num_col, mask) in enumerate(
            zip(questions, headers, table_words, labels, all_num_cols, masks)):
        a, b, c, d, e, f = np.array(question), np.array(header), np.array(table_word), np.array(label), np.array(
            all_num_col), np.array(mask)
        loss = train_step(a, b, c, d, e, f)
        if index % 100 == 0:
            print('Done ' + str(index) + ' train iterations out of ' + str(train_iterations) + ' Loss is ' + str(
                float(loss.numpy())))


def test_query_encoder(batches, query_embedding_step):
    sample_info_dict = {}
    for test_batch in batches:
        questions = [x['question_tokens'] for x in test_batch]
        embeddings = query_embedding_step(np.array(questions)).numpy()
        for info, embedding in zip(test_batch, embeddings):
            sample_info_dict[info['question_tokens']] = {'table_id': info['table_id'], 'embedding': embedding}
    return sample_info_dict


def test_table_encoder(headers, table_words, all_num_cols, masks, table_ids, table_embedding_step):
    index = 0
    index_to_vec, table_id_to_index = {}, {}
    for a, b, c, d, e in zip(headers, table_words, all_num_cols, masks, table_ids):
        table_encodings = table_embedding_step(np.array(a), np.array(b), np.array(c),
                                               np.array(d, dtype=np.float32)).numpy()
        for table_id, encoding in zip(e, table_encodings):
            index_to_vec[index] = encoding
            table_id_to_index[table_id] = index
            index = index + 1
    return index_to_vec, table_id_to_index


def get_metrics(ranks):
    dp = [0 for _ in range(max(ranks) + 1)]
    rr = 0
    for rank in ranks:
        dp[rank] = dp[rank] + 1
        rr = rr + 1 / float(rank)
    mrr = rr / len(ranks)
    for index in range(1, len(dp)):
        dp[index] = dp[index] + dp[index - 1]
    total = dp[-1]
    p_scores = []
    for num in dp:
        p_scores.append(num / float(total))
    return mrr, p_scores


def metrics_logger(sample_info_dict, id_to_index, index_file, dim, eval_set, find_p, log_file):
    ranks = get_query_table_ranks(sample_info_dict, id_to_index, index_file, dim)
    mrr, p_scores = get_metrics(ranks)
    log_file.write(eval_set + '\t' + str(mrr))
    for p in find_p:
        log_file.write('\t' + str(p_scores[p]))
    log_file.write('\n')


def baseline_metrics_logger(dev_ranks, test_ranks, log_file, find_p):
    mrr_dev, p_scores_dev = get_metrics(dev_ranks)
    mrr_test, p_scores_test = get_metrics(test_ranks)
    log_file.write('dev' + '\t' + str(mrr_dev))
    for p in find_p:
        log_file.write('\t' + str(p_scores_dev[p]))
    log_file.write('\n')
    log_file.write('test' + '\t' + str(mrr_test))
    for p in find_p:
        log_file.write('\t' + str(p_scores_test[p]))
    log_file.write('\n')
