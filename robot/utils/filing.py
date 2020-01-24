import os
import sys
import datetime
import string


class FilingSystem:

    def __init__(self, experiment):

        self.exp = experiment

        self.exp_class_folder1 = os.path.join(self.exp.local_root, self.exp.experiment_name)
        self.exp_class_folder2 = os.path.join(self.exp.remote_root, self.exp.experiment_name)
        self.get_unique_identifier_and_folder()


    def get_unique_identifier_and_folder(self):
        now = datetime.datetime.now()
        today = now.strftime('%y%m%d')

        tag_index = 0
        unique_tags = string.ascii_lowercase
        self.unique_identifier = '{}{}'.format(today, unique_tags[tag_index])
        self.local_exp_folder = os.path.join(self.exp_class_folder1, self.unique_identifier)

        while os.path.exists(self.local_exp_folder):
            if tag_index == len(unique_tags) - 1:
                print("No more unique fullcode tags available today. Exiting.")
                sys.exit()
            self.unique_identifier = '{}{}'.format(today, unique_tags[tag_index])
            self.local_exp_folder = os.path.join(self.exp_class_folder1, self.unique_identifier)
            tag_index += 1
        self.remote_exp_folder = os.path.join(self.exp_class_folder2, self.unique_identifier)

    def prepare_image_filing(self, reaction_num, image_num):
        rxn_str = str(reaction_num+1).zfill(len(str(self.exp.number_of_reactions)))
        img_str = str(image_num+1).zfill(len(str(self.exp.images_per_reaction)))

        local_rxn_dir = os.path.join(self.local_exp_folder, 'reaction_{}'.format(rxn_str))
        local_img_dir = os.path.join(local_rxn_dir, 'Images')
        local_img_path = os.path.join(local_img_dir, 'Image_{}.png'.format(img_str))

        remote_rxn_dir = os.path.join(self.remote_exp_folder, 'reaction_{}'.format(rxn_str))
        remote_img_dir = os.path.join(remote_rxn_dir, 'Images')
        remote_img_path = os.path.join(remote_img_dir, 'Image_{}.png'.format(img_str))

        os.makedirs(local_img_dir, exist_ok=True)
        os.makedirs(remote_img_dir, exist_ok=True)

        return local_img_path, remote_img_path

    def prepare_masks_filing(self, reaction_num, image_num):
        rxn_str = str(reaction_num+1).zfill(len(str(self.exp.number_of_reactions)))
        img_str = str(image_num+1).zfill(len(str(self.exp.images_per_reaction)))

        local_rxn_dir = os.path.join(self.local_exp_folder, 'reaction_{}'.format(rxn_str))
        local_msk_dir = os.path.join(local_rxn_dir, 'masks')
        local_msk_dir = os.path.join(local_msk_dir, 'Image_{}'.format(img_str))

        remote_rxn_dir = os.path.join(self.remote_exp_folder, 'reaction_{}'.format(rxn_str))
        remote_msk_dir = os.path.join(remote_rxn_dir, 'masks')
        remote_msk_dir = os.path.join(remote_msk_dir, 'Image_{}'.format(img_str))

        os.makedirs(local_msk_dir, exist_ok=True)
        os.makedirs(remote_msk_dir, exist_ok=True)

        return local_msk_dir, remote_msk_dir

    def prepare_vial_filing(self, reaction_num, image_num):
        rxn_str = str(reaction_num+1).zfill(len(str(self.exp.number_of_reactions)))
        img_str = str(image_num+1).zfill(len(str(self.exp.images_per_reaction)))

        local_rxn_dir = os.path.join(self.local_exp_folder, 'reaction_{}'.format(rxn_str))
        local_vial_dir = os.path.join(local_rxn_dir, 'vials')
        local_vial_path = os.path.join(local_vial_dir, 'Image_{}.txt'.format(img_str))

        remote_rxn_dir = os.path.join(self.remote_exp_folder, 'reaction_{}'.format(rxn_str))
        remote_vial_dir = os.path.join(remote_rxn_dir, 'vials')
        remote_vial_path = os.path.join(remote_vial_dir, 'Image_{}.txt'.format(img_str))

        os.makedirs(local_vial_dir, exist_ok=True)
        os.makedirs(remote_vial_dir, exist_ok=True)

        return local_vial_path, remote_vial_path

    def prepare_string_filing(self, reaction_num, image_num):
        rxn_str = str(reaction_num+1).zfill(len(str(self.exp.number_of_reactions)))
        img_str = str(image_num+1).zfill(len(str(self.exp.images_per_reaction)))

        local_rxn_dir = os.path.join(self.local_exp_folder, 'reaction_{}'.format(rxn_str))
        local_vial_dir = os.path.join(local_rxn_dir, 'strings')
        local_vial_path = os.path.join(local_vial_dir, 'Image_{}.txt'.format(img_str))

        remote_rxn_dir = os.path.join(self.remote_exp_folder, 'reaction_{}'.format(rxn_str))
        remote_vial_dir = os.path.join(remote_rxn_dir, 'strings')
        remote_vial_path = os.path.join(remote_vial_dir, 'Image_{}.txt'.format(img_str))

        os.makedirs(local_vial_dir, exist_ok=True)
        os.makedirs(remote_vial_dir, exist_ok=True)

        return local_string_path, remote_string_path
