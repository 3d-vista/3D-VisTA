import csv

class LabelConverter(object):
    def __init__(self, file_path):
        self.raw_name_to_id = {}
        self.nyu40id_to_id = {}
        self.nyu40_name_to_id = {}
        self.scannet_name_to_scannet_id = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.id_to_scannetid = {}
        with open(file_path) as fd:
            rd = list(csv.reader(fd, delimiter="\t", quotechar='"'))
            for i in range(1, len(rd)):
                raw_id = i - 1
                raw_name = rd[i][1]
                nyu40_id = int(rd[i][4])
                nyu40_name = rd[i][7]
                
                self.raw_name_to_id[raw_name] = raw_id
                self.nyu40id_to_id[nyu40_id] = raw_id
                self.nyu40_name_to_id[nyu40_name] = raw_id
                if nyu40_name not in self.scannet_name_to_scannet_id.keys():
                    self.id_to_scannetid[raw_id] = self.scannet_name_to_scannet_id['others']
                else:
                    self.id_to_scannetid[raw_id] = self.scannet_name_to_scannet_id[nyu40_name]
        