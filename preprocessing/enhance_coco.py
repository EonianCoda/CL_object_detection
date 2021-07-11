from collections import defaultdict
import pandas as pd
from pycocotools.coco import COCO

class Enhance_COCO(COCO):
    def __init__(self, path):

        super().__init__(path)
        self.classes = defaultdict()
        self.reverse_classes = defaultdict()
        for category in self.loadCats(self.getCatIds()):
            self.classes[category['id']] = category['name']
            self.reverse_classes[category['name']] = category['id']

    def get_cats_by_imgs(self, imgIds, return_name=False):
        """ Given some image ids, and return the category in these imgs

            Args:
                imgIds: the image ids
                return_name: return the name of category or the index of category, when return_name = 'true', return names, else return indexs 
        """
        annotations = self.loadAnns(self.getAnnIds(imgIds=imgIds)) # get annotations
        #get categoryId appear in annotations
        catIds = [ann['category_id'] for ann in annotations]
        catIds = list(set(catIds))

        if not return_name:
            return catIds
        else:
            catNames = []
            #get category name from categord Id
            for catId in catIds:
                catNames.append(self.classes[catId])
            return catNames

    def get_imgs_by_cats(self, catIds):
        """ Given some indexs of category, and return all imgs having these category

            Args:
                catIds: the index of category
        """
        if type(catIds) == list:
            imgIds = set()
            for catId in catIds:
                imgIds.update(self.getImgIds(catIds=catId))
            return list(imgIds)
        else:
            return self.getImgIds(catIds=catIds)

    def catId_to_name(self, catIds):
        """Convert the indexs of categories to the name of them
            Args:
                catIds: a list of int or int, indicating the index of categories
            Return: a list contains the name of category which is given
        """
        if type(catIds) == int:
            return [self.classes[catIds]]
        else:
            names = [self.classes[catId] for catId in catIds]
            return names
                
    def catName_to_id(self, names, sort = True):
        """Convert the names of categories to the index of them
            Args:
                names: a list of str or str, indicating the nameof categories
                sort: whether the list which is returned is well-sorted, default = True
            Return: a list contains the indexs of category given
        """
        if isinstance(names, str):
            return [self.reverse_classes[names]]

        names = list(set(names))
        
        ids = []
        for name in names:
            ids.append(self.reverse_classes[name])
        if sort:
            ids.sort()

        return ids
    
    def get_catNum_by_catId(self, catIds):
        result = {'image':[], 'object':[]}
        index = []
        catIds.sort()
        
        for catId in catIds:
            index.append(self.classes[catId])
            result['image'].append(len(self.getImgIds(catIds = catId)))
            result['object'].append(len(self.getAnnIds(catIds = catId)))

        index.append('Counts')
        result['image'].append(sum(result['image']))
        result['object'].append(sum(result['object']))
        result = pd.DataFrame(result, index=index)
        result.sort_values(by=['image'], ascending=False)
        return result

    def get_catNum_by_imgs(self, imgIds):
        #get annotations
        annotations = self.loadAnns(self.getAnnIds(imgIds=imgIds))
        #get categoryId appear in annotations
        catIds = [ann['category_id'] for ann in annotations]

        result = {'image':[], 'object':[]}

        index = self.catId_to_name(list(set(catIds)))
        #object counts
        result['object'] = np.unique(catIds, return_counts=True)[1].tolist()

        catIds = list(set(catIds))
        for catId in catIds:
            result['image'].append(len(set(self.getImgIds(catIds = catId)) & set(imgIds)))
        
        print('Counts meaning for  image is your input imgIds number')
        index.append('Counts')
        result['image'].append(len(imgIds))
        result['object'].append(sum(result['object']))
        result = pd.DataFrame(result, index=index)
        result.sort_values(by=['image'], ascending=False)
        return result
