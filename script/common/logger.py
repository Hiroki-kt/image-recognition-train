# -*- coding: utf-8 -*-
from django.conf import settings
if ( settings.LOGGER_TYPE == "celery" ):
    from celery.utils.log import get_task_logger
else:
    import logging

def get_task_logger(log_name):
    return logging.getLogger(log_name)
