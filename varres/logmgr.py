import logging

def logger(name,level='INFO'):
    logger = logging.getLogger(name)
    if level in 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if (len(logger.handlers) == 0):
        # create console handler and set level to debug
        ch = logging.StreamHandler()

        if level in 'DEBUG':
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(funcName)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        if name == 'varres':
            logger.info('varrespy - v 1.0')
            logger.info('----------------')
    else:
        ch = logger.handlers[0]
        if level in 'DEBUG':
            ch.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)
            logger.setLevel(logging.DEBUG)


    return logger

############################################################
# Program is part of varres                                #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
