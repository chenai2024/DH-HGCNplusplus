# For Yelp dataset
from base.deepRecommender import DeepRecommender
from base.socialRecommender import SocialRecommender
import sys
from base.multipleRelation import buildItemRelation
from base.multipleRelation import multipleRelation
import os
from os.path import abspath
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
from util import config

sys.path.append('model')

class DH_HGCNplusplus2(SocialRecommender, DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,
                                   fold=fold)

    def readConfiguration(self):
        super(DH_HGCNplusplus2, self).readConfiguration()
        args = config.LineConfig(self.config['DH_HGCNplusplus2'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])

    def buildSparseFRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        FAdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        return FAdjacencyMatrix

    def buildSparseCRelationMatrix(self):
        row, col, entries = [], [], []
        a = multipleRelation()
        commentrelation = a.buildCommentRelation(abspath(self.config['comment']))
        for pair in commentrelation:
            try:
                if pair[0] in self.data.user.keys() and pair[1] in self.data.user.keys():
                    row += [self.data.user[pair[0]]]
                    col += [self.data.user[pair[1]]]
                    entries += [1.0]
            except:
                pass
        CAdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        return CAdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def getItemMatrix(self):
        row, col, entries = [], [], []
        itemclassify = buildItemRelation(abspath(self.config['itemsim']))
        for pair in itemclassify:
            try:
                if pair[0] in self.data.item.keys():
                    row += [self.data.item[pair[0]]]
                    col += [pair[1]]
                    entries += [1.0]
            except:
                pass
        labels = int(self.config['K'])
        item_AdjMatrix = coo_matrix((entries, (row, col)), shape=(self.num_items, labels),dtype=np.float32)
        return item_AdjMatrix

    def buildMotifInduceAdjacencyMatrix(self):
        F = self.buildSparseFRelationMatrix()
        C = self.buildSparseCRelationMatrix()
        R = self.buildSparseRatingMatrix()
        self.userAdjacency = R.tocsr()
        self.itemAdjacency = R.T.tocsr()
        # FRIEND
        B = F.multiply(F.T)
        U = F - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        # COMMENT
        B1=C.multiply(C.T)
        A9 = (B1.dot(B1)).multiply(B1)
        # F
        A10 = (R.dot(R.T)).multiply(B)
        A11 = (R.dot(R.T)).multiply(U)
        A11 = A11 + A11.T
        # C
        A13 = (R.dot(R.T)).multiply(B1)
        A14 = R.dot(R.T) - A10 - A11 - A13
        # ITEM
        A15 = (R.T).dot(R)

        H_f = sum([A1, A2, A3, A4, A5, A6, A7])
        H_f = H_f.multiply(1.0 / H_f.sum(axis=1).reshape(-1, 1))

        H_c = A9
        H_c = H_c.multiply(1.0 / H_c.sum(axis=1).reshape(-1, 1))
        H_p = sum([A10, A11, A13, A14])
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / H_p.sum(axis=1).reshape(-1, 1))

        # ITEM
        H_item = A15
        H_item = H_item.multiply(H_item > 1)
        H_item = H_item.multiply(1.0 / H_item.sum(axis=1).reshape(-1, 1))

        return [H_f, H_c, H_p, H_item]


    def generate_G_from_H(self,H):
        # calculate G from hypgraph incidence matrix H
        # D−v1/2 H D−e 1HT D−v 1/2
        # :param H: hypergraph incidence matrix H
        # :return G
        DV = H.sum(axis=1).reshape(1, -1)
        DE = H.sum(axis=0).reshape(1, -1)
        temp1 = (H.transpose().multiply(np.sqrt(1.0 / DV))).transpose()
        temp2 = temp1.transpose()
        G = temp1.multiply(1.0 / DE).dot(temp2)
        G = G.tocoo()
        indices = np.mat([G.row, G.col]).transpose()
        G = tf.SparseTensor(indices, G.data.astype(np.float32), G.shape)
        return G

    def adj_to_sparse_tensor(self, adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    def initModel(self):
        super(DH_HGCNplusplus2, self).initModel()
        total=self.buildMotifInduceAdjacencyMatrix()
        H_f=total[0]
        H_f = self.adj_to_sparse_tensor(H_f)
        H_c = total[1]
        H_c = self.adj_to_sparse_tensor(H_c)
        H_p = total[2]
        H_p = self.adj_to_sparse_tensor(H_p)
        H_item = total[3]
        H_item = self.adj_to_sparse_tensor(H_item)
        H_sim = self.getItemMatrix()
        H_sim = self.generate_G_from_H(H_sim)

        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_channel = 3
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        for i in range(self.n_channel):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.embed_size, self.embed_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.embed_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.embed_size, self.embed_size]),
                                                              name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.embed_size]),
                                                                   name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.embed_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.embed_size, self.embed_size]), name='atm')

        # function define
        def self_gating(em, channel):
            return tf.multiply(em, tf.nn.sigmoid(
                tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))

        def channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(
                    tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(
                    tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings, score

        user_embeddings_c1 = self_gating(self.user_embeddings, 1)
        user_embeddings_c2 = self_gating(self.user_embeddings, 2)
        interaction_user_embeddings = self_gating(self.user_embeddings, 3)

        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embedding_interaction = [interaction_user_embeddings]

        'item-embedding'
        item_embeddings = self.item_embeddings
        all_item_embeddings_fea = [item_embeddings]
        all_item_embeddings_rating = [item_embeddings]

        self.ss_loss = 0
        for k in range(self.n_layers):
            # S-F
            user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_f, user_embeddings_c1)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            all_embeddings_c1 += [norm_embeddings]
            # S-C
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_c, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            # S-interaction
            interaction_user_embeddings = tf.sparse_tensor_dense_matmul(H_p, interaction_user_embeddings)
            all_embedding_interaction += [tf.math.l2_normalize(interaction_user_embeddings, axis=1)]
            # item
            dims = list(H_sim.shape)
            d1 = int(dims[1])
            d2 = self.embed_size
            temp = tf.Variable(tf.truncated_normal(shape=[d1, d2], stddev=0.1))
            item_fea = tf.sparse_tensor_dense_matmul(H_sim, temp)
            norm_embeddings = tf.math.l2_normalize(item_fea, axis=1)
            all_item_embeddings_fea += [norm_embeddings]

            item_rating = tf.sparse_tensor_dense_matmul(H_item,item_embeddings)
            norm_embeddings = tf.math.l2_normalize(item_rating, axis=1)
            all_item_embeddings_rating += [norm_embeddings]

        # averaging user embeddings
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        interaction_user_embeddings = tf.reduce_sum(all_embedding_interaction, axis=0)
        item_embeddings_fea = tf.reduce_sum(all_item_embeddings_fea, axis=0)
        item_embeddings_rating = tf.reduce_sum(all_item_embeddings_rating, axis=0)
        new_item_embeddings = item_embeddings_fea + item_embeddings_rating
        # aggregating
        self.final_item_embeddings = new_item_embeddings
        self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c1,user_embeddings_c2)
        self.final_user_embeddings += interaction_user_embeddings / 2

        # embedding look-up
        self.neg_item_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)

        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.final_item_embeddings), 1)
        self.cnt_loss = self.cal_contrastive_loss(item_embeddings_fea, item_embeddings_rating)
        self.cnt_loss += self.cal_contrastive_loss(user_embeddings_c1, interaction_user_embeddings)
        self.cnt_loss += self.cal_contrastive_loss(user_embeddings_c2, interaction_user_embeddings)


    def cal_contrastive_loss(self,pos,emb):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
        pos_score_item = tf.reduce_sum(tf.multiply(pos,emb), axis=1)
        ttl_score_item = tf.matmul(pos, row_shuffle(emb), transpose_a=False,transpose_b=True)
        pos_score = tf.exp(pos_score_item / 0.2)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score_item / 0.2),axis=1)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        return ssl_loss


    def buildModel(self):
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001 * tf.nn.l2_loss(self.weights[key])
        rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + \
                   self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss + reg_loss +self.ss_rate * self.cnt_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        pltloss = []
        for iteration in range(self.maxIter):
            epochloss = 0
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l1 = self.sess.run([train_op, rec_loss],
                                      feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('[', self.foldInfo, ']', 'training:', iteration + 1, 'batch', n, 'rec loss:',
                      l1)
                epochloss += l1
            pltloss.append(epochloss / n)
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
