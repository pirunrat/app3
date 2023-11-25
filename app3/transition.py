import sys, os
sys.path.append('app3')
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
from django.conf import settings
from utils import Utility                                                                                                                

if __name__ == "__main__":
    util = Utility()
    util.register_model_to_production()