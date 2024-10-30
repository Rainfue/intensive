# %%
import cianparser as cian
from time import sleep
# %%
towns = [ 'Сергиев Посад',  'Ивантеевка','Бронницы', 'Голицыно','Луховицы']
# %%
for town in towns:
  town_parser = cian.CianParser(location=town)
  rooms = [1, 2, 3, 4, 5, 'studio']
  for room in rooms:
    data = town_parser.get_flats(
      deal_type='sale',
      rooms=room,
      with_saving_csv=True,
      additional_settings={"start_page":1, "end_page":25},
      with_extra_data=True)
    sleep(30)
sleep(60)
# %%
cian.list_locations()
# %%
