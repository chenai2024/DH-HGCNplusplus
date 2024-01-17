from collections import defaultdict


class multipleRelation():

    def __init__(self):
        self.user = {}
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)

    def buildCommentRelation(self, fin):
        commentrelation=[]
        triple=[]
        with open(fin, 'r', encoding='utf-8') as f:
            commentrelations=f.readlines()
            for index, info in enumerate(commentrelations):
                users=info.strip('\n').split()
                userId1=users[0]
                userId2=users[1]
                weight=1
                commentrelation.append([userId1, userId2, weight])
        for line in commentrelation:
            userId1, userId2, weight = line
            self.followees[userId1][userId2] = weight
            self.followers[userId2][userId1] = weight
            if userId1 not in self.user:
                self.user[userId1] = len(self.user)
            if userId2 not in self.user:
                self.user[userId2] = len(self.user)
            triple.append([self.user[userId1], self.user[userId2], weight])
        return commentrelation

    def buildGroupRelation(self,fin):
        grouprelation = []
        triple = []
        with open(fin, 'r', encoding='utf-8') as f:
            read = f.readlines()
            for index, info in enumerate(read):
                users = info.strip('\n').split()
                userId1 = users[0]
                userId2 = users[1]
                weight = 1
                grouprelation.append([userId1, userId2, weight])
        for line in grouprelation:
            userId1, userId2, weight = line
            self.followees[userId1][userId2] = weight
            self.followers[userId2][userId1] = weight
            if userId1 not in self.user:
                self.user[userId1] = len(self.user)
            if userId2 not in self.user:
                self.user[userId2] = len(self.user)
            triple.append([self.user[userId1], self.user[userId2], weight])
        return grouprelation

    def row(self,u):
        return self.commentMatrix.row(self.user[u])

    def col(self,u):
        return self.commentMatrix.col(self.user[u])

    def elem(self,u1,u2):
        return self.commentMatrix.elem(u1,u2)

    def weight(self,u1,u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    def trustSize(self):
        return self.commentMatrix.size

    def getFollowers(self,u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def getFollowees(self,u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def hasFollowee(self,u1,u2):
        if u1 in self.followees:
            if u2 in self.followees[u1]:
                return True
            else:
                return False
        return False

    def hasFollower(self,u1,u2):
        if u1 in self.followers:
            if u2 in self.followers[u1]:
                return True
            else:
                return False
        return False

# item relation
def buildItemRelation(fin):
    itemrelation = []
    with open(fin, 'r', encoding='utf-8') as f:
        read = f.readlines()
        for index, info in enumerate(read):
            info = info.strip('\n').split()
            itemid = info[0]
            label = info[1]
            weight = 1
            itemrelation.append([itemid, label, weight])
    return itemrelation