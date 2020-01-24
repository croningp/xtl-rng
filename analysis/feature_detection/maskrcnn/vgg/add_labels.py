import json
import sys
import os

# object_json_file = r'Z:\group\Edward Lee\03-Projects\07-RandomMOF\code\vgg\images\img_20190705_161441\5,5\via_region_data.json'


folder = sys.argv[1]

object_json_file = os.path.join(folder, 'via_region_data.json')

data = json.load(open(object_json_file))

region_attribute_dict = {'object_name':'crystal'}
new_data = {}

for k, v in data.items():
    for idx, region in enumerate(v['regions']):
        data[k]['regions'][idx]['region_attributes'] = region_attribute_dict


with open(object_json_file, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)