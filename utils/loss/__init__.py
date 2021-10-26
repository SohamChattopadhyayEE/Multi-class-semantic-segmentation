import logging
import functools

from utils.loss.loss import (cross_entropy2d, 
                soft_dice_loss, bootstrapped_cross_entropy2d,
                multi_scale_cross_entropy2d)


logger = logging.getLogger("utils")

dict_loss = {
    "soft_dice_loss" : soft_dice_loss,
    "cross_entropy2d" : cross_entropy2d,
    "bootstrapped_cross_entropy2d" : bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy2d" : multi_scale_cross_entropy2d
}

def get_loss_function(cfg = "cross_entropy2d"):
    if cfg is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        '''
        loss_dict = cfg
        loss_name = cfg#loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in dict_loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        '''
        loss_name = cfg
        if loss_name not in dict_loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))
        return dict_loss[loss_name] #functools.partial(dict_loss[loss_name], **loss_params)