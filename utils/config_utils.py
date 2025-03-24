import os
import streamlit as st
from ruamel.yaml import YAML
yaml = YAML()

def get_default_config(project_path, project_name, **kwargs):
    config_dir = os.path.join(project_path, 'config.yml') # save the config file under the root of the project


    defaut_config = {

        'Project_folders':
        {
        'Task': project_name,
        'project_path': project_path,
        'model_path': os.path.join(project_path, 'models'),
        'output_path': os.path.join(project_path, 'output'),
        'data_path': os.path.join(project_path, 'datasets'),
        'label_path': os.path.join(project_path, "label"),
        'sample_path': os.path.join(project_path, 'sample'),
        'video_path': os.path.join(project_path, 'videos'),
        'train_videolist': os.path.join(project_path, 'videos', 'video_segments_names.text'),
            'config_path': os.path.join(project_path, 'config.yml')
             },

        'Dataset':

        {'train': None,
        'test': None,
        'is_single_action': False,
        'multi_action_crop': 10,
        'single_action_crop': 10,
         },

        # Training,Evaluation and Analysis configuration
        'Training': {

        'TrainingFraction': None,
        'iteration': None,
        'default_net_type': None,
        'default_augmenter': None,
        'snapshotindex': None,
        # Training setting
        'batch_size': 64,
        'learning_rate': 0.0001,
        'loss_type': 'L1',
        'un_epoch': 30,
        'su_epoch': 30,
        'device': 'cuda',
        },

        # Model Parameters
        'Model':

            {'feature_length': 16,
        'hidden_size': 215,
        'en_num_layers': 3,
        'de_num_layers': 1,
        'cla_num_layers': 1,
        'cla_dim': 8,
        'num_class': 8,
        'teacher_force': False,
        'fix_weight': False,
        'fix_state': False,

            # Iter training setting
        'sample_per': None,
        'iter_times': None,
        'iter_epoch':None,
        'labeled_id': None,
        'display_iters': 3,
        'save_iters': 30,
        'multi_epoch': 30,
        'tr_modelType': 'seq2seq',
        'tr_modelName':None,
        'sample_method':'Marginal Index (MI)',
        'label_budget': 10,
         'class_name': ['drink',
                        'eat',
                        'groom',
                        'hang',
                        'head',
                        'rear',
                        'rest',
                        'walk', ],
             },


            # Video Name (Ordered as as in dataset)

            # class name the order will be the same as label start from 1 to N
            # make sure to change "cla_dim" and "num_class" if you change number of classes
    }

    for key in kwargs:
        for groups in ['Project_folders', 'Dataset', 'Training', 'Model']:
            if key in defaut_config[groups].keys():
                defaut_config[groups][key] = kwargs[key]
                break
    st.session_state.config_dict = defaut_config
    try:
        with open(config_dir, 'w') as f:
            yaml.dump(defaut_config, f)
    except:
        ValueError('Enter all the information to continue')

    for subfolder_key in ['model_path',
                        'output_path',
                        'data_path',
                        'label_path',
                        'sample_path',
                          'video_path']:
        subfolder = defaut_config['Project_folders'][subfolder_key]
        print(subfolder_key)
        print(subfolder)
        if not os.path.exists(subfolder):
            try:
                os.mkdir(subfolder)
            except:
                FileExistsError('Enter data directory first.')


def load_cfg_file(cfg_file_buffer):
    #with open(cfg_file_dir) as f:
    config= yaml.load(cfg_file_buffer)

    st.session_state.config_dict = config

def revise_cfg_file(config, **kwargs):
    # with open(cfg_file_dir) as f:
    #     config = yaml.load(f)
    #
    config_file_dir = config['Project_folders']['config_path']
    for key in kwargs:
        for groups in ['Project_folders', 'Dataset', 'Training', 'Model']:
            if key in config[groups].keys():
                config[groups][key] = kwargs[key]
                break

    with open(config_file_dir, 'w') as f:
        yaml.dump(config, f)

    st.session_state.config_dict = config
