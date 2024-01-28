import h5py
import joblib


def load_collections(attr_ids):
    if type(attr_ids) is not list:
        attr_ids = [attr_ids]
    
    collections_id = []
    collections = []
     
    for attr_id in attr_ids:
        collection_id = joblib.load("collections/c_idxs_" + str(attr_id) + ".joblib")
        collection = joblib.load("collections/c_feats_" + str(attr_id) + ".joblib")
        collections_id.append(collection_id)
        collections.append(collection)
         
    return collections_id, collections


def collections_to_h5py(n_attrs=8):
    collections_id, collections = load_collections([0, 1, 2, 3, 4, 5, 6, 7])
    
    with h5py.File("collections/data.h5", 'w') as file:
        for i in range(n_attrs): 
            file.create_dataset('attr' + str(i), data=collections[i])
            
collections_to_h5py()