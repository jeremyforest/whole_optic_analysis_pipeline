import json
import numpy as np
import matplotlib.pyplot as plt
import sys



with open ('/media/jeremy/Data/local/Data_manip/2020_01_24/experiment_10/experiment_10_timings.json') as file:   ## camera and dlp, real expe
    timings_data = dict(json.load(file))
timings_dlp_on = timings_data['dlp']['on']
timings_dlp_off = timings_data['dlp']['off']
timings_laser_on = timings_data['laser']['on']
timings_laser_off = timings_data['laser']['off']
timings_camera_images = []
for images_timings in timings_data['camera']:
    for image_timing in images_timings:
        timings_camera_images.append(image_timing)


# print(timings_camera_images[0:10])

timings_camera_images = np.array(timings_camera_images)*(10**9)
timings_camera_images[0:10]

timings_camera_images = np.sort(timings_camera_images)
c = timings_camera_images
d = range(len(timings_camera_images))
plt.plot(d,c)

timing_difference = [(timings_camera_images[j+1] - timings_camera_images[j]) for j in range(len(timings_camera_images)-1)]
a = timing_difference
b = range(len(timings_camera_images)-1)
plt.plot(b,a)

timing_difference = np.array(timing_difference)/1000000
np.set_printoptions(threshold=sys.maxsize)
print(timing_difference)


weird = timing_difference[timing_difference > 50]
weird_index = [(np.where(timing_difference == index)[0]) for index in weird]
weird_index_difference = [(weird_index[k+1][0] - weird_index[k][0]) for k in range(len(weird_index)-1)]
print(weird_index_difference)

timings_camera_images[390:410]
timing_difference[390:410]
