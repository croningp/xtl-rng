import time
from operator import itemgetter

def order_reagents(rxn_reagents):

    """ orders reagents based on dispensation time """

    new_list = {}
    for k, v in rxn_reagents.items():
        new_list[k] = v['time']
    new_list = sorted(new_list.items(), key = itemgetter(1))

    ordered_reagents = []
    for item in new_list:
        ordered_reagents.append([item[0],rxn_reagents[item[0]]])
    return ordered_reagents

def get_updates(rxn_number, conf):
    """ determine when each image should be taken """
    updates = []
    for image_idx in range(1,conf['images_per_reaction']+1):
        image_time = time.time()+(conf['number_of_reactions']*5) + (image_idx*conf['time_between_images'])
        image_op = {'type': 'image', 'time':image_time, 'details': {'rxn_index':rxn_number, 'img_index':image_idx}}
        updates.append(image_op)
    return updates
