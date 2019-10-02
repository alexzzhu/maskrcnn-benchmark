# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch

from maskrcnn_benchmark.utils.imports import import_file


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    #log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    #logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if not model_state_dict[key].shape == loaded_state_dict[key_old].shape:
            continue
        
        model_state_dict[key] = loaded_state_dict[key_old]
        #logger.info(
        #    log_str_template.format(
        #        key,
        #        max_size,
        #        key_old,
        #        max_size_loaded,
        #        tuple(loaded_state_dict[key_old].shape),
        #    )
        #)
    

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

# https://code.activestate.com/recipes/577346-getattr-with-arbitrary-depth/
def multi_getattr(obj, attr, default = None):
    """
    Get a named attribute from an object; multi_getattr(x, 'a.b.c.d') is
    equivalent to x.a.b.c.d. When a default argument is given, it is
    returned when any attribute in the chain doesn't exist; without
    it, an exception is raised when a missing attribute is encountered.

    """
    attributes = attr.split(".")
    for i in attributes:
        try:
            obj = getattr(obj, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return obj

def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    is_event = False
    if model_state_dict['backbone.body.stem.conv1.weight'].shape[1] != 3:
        is_event = True
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")

    if is_event and not 'backbone.body.stem.conv1.weight' in loaded_state_dict:
        print("First time restoring weights, setting stem to random initialization.")
        strict = False
        deleted_keys = [ x for x in model_state_dict if 'stem' in x ]
        model_state_dict = { x : model_state_dict[x] for x in model_state_dict if not 'stem' in x }

        for key in deleted_keys:
            var = multi_getattr(model, key)
            if 'conv1.weight' in key:
                torch.nn.init.normal_(var, std=.01)
            elif 'bn1.weight' in key:
                torch.nn.init.uniform_(var)
            elif 'bn1.bias' in key:
                torch.nn.init.zeros_(var)
    else:
        print("Previous correct stem weights found, using these.")        
        strict = True
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict, strict=strict)
