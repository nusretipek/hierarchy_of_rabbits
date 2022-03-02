#import pandas as pd
#pd.set_option('display.width', 320)
#pd.set_option('display.max_columns',15)


#file = open('/home/nipek/log.txt', "r")
#list_of_lines = file.readlines()
#file.close()

#for i in range(len(list_of_lines)):
#    comment_index = list_of_lines[i].find('Comment:')
#    list_of_lines[i] = list_of_lines[i][:comment_index+9] + list_of_lines[i][comment_index+9:].replace(' ', '_')

#file = open('/home/nipek/log2.txt', 'w')
#file.writelines(list_of_lines)
#file.close()

#df = pd.read_csv('/home/nipek/log2.txt', delim_whitespace = True, header = None)
#df = df.drop([2, 4, 5, 7, 9, 11 , 13], axis=1)
#for col in [6, 8, 10, 12, 14]: df[col] = df[col].apply(lambda x: x.replace(';', ''))
#df.columns = ['Date', 'Time', 'Type', 'Thread ID', 'Class', 'Method', 'Line', 'Comment']
#df['Comment'] = df.Comment.apply(lambda  x: x.replace('_', ' '))
#print(df)

#import geopy
#import pandas as pd
#from geopy.geocoders import Nominatim
#geolocator = Nominatim(user_agent="StackOverFlow", timeout=3)
#geo_location = geolocator.geocode(loc, addressdetails=True, language='en')
#print(geo_location.raw["lat"], geo_location.raw["lon"], geo_location.raw["address"]['city'])

#df = pd.DataFrame({'locations': ['King Saud Road, Al Khobar 31952 Saudi Arabia',
#                            '123 Main Street, Los Angeles, CA 90034, USA']})
#df['location_info'] = df.locations.apply(lambda x: geolocator.geocode(x, addressdetails=True, language='en'))
#df['latitude'] = df.location_info.apply(lambda x: x.raw['lat'])
#df['longitude'] = df.location_info.apply(lambda x: x.raw['lon'])
#df['city'] = df.location_info.apply(lambda x: x.raw["address"]['city'])
#df = df.drop(['location_info'], axis=1)
#print(df)

#import sys
#import zlib
#import base64
#text = "THE QUICK BROWN FOX JUMPS OVER A LAZY DOG."
#print(sys.getsizeof(zlib.compress(text.encode('utf16')))) -> 104 Bytes
#print(sys.getsizeof(zlib.compress(base64.b64encode(text.encode('utf16'))))) - 127 Bytes
#print(sys.getsizeof(base64.b64encode(zlib.compress(text.encode('utf16'))))) - 129 Bytes

import re
import json
import pandas as pd

with open('/home/nipek/mM6XxeRq.txt') as f:
    full_file = f.read()
    full_file = '{' + full_file[1:-1] + '}'
    json_file = json.loads(full_file)
f.close()
print(json_file)
    #names = (x['name'] for x in json.loads(line))
        #print(names)


