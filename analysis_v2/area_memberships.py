import pandas as pd
from microns_phase3 import nda, utils

areas = nda.AreaMembership.fetch(format='frame').reset_index()
areas.to_csv('area_memberships.csv')