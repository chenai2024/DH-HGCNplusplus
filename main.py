from QRec import QRec
from util.config import Config
if __name__ == '__main__':

    print('='*80)
    print('='*80)
    print('1. DH_HGCN++         2. DH_HGCN++2')

    print('='*80)
    num = input('please enter the number of the matrixDefine you want to run:')
    import time
    s = time.time()
    models = {'1':'DH_HGCNplusplus','2':'DH_HGCNplusplus2'}
    try:
        conf = Config('./config/'+models[num]+'.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = QRec(conf)
    recSys.execute()
    e = time.time()
    print("Run time: %f s" % (e - s))
