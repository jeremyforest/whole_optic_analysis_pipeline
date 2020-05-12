import numpy as np
import plotly.graph_objects as go
import argparse
import pdb
import copy

# parser = argparse.ArgumentParser(description="options")
# parser.add_argument("--path", required=True, action="store",
#                         help="main folder absolute path, something like /media/jeremy/Data/local/Data_manip/2020_02_05")
# parser.add_argument("--experiments", required=True, action="append", type=int,
#                         help="values of experiments to go over")


normalize = True

path = '/media/jeremy/Data/local/Data_manip/2020_01_24/'

## only AP ?
# experiments = [252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270]
# manual_laser_frame_on = [788,786,788,784,782,770,764,785,784,789,784,786,778,786,783,785,780,786,772]  ## only good indication for legacy data

#
# experiments =           [364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462]       ## 462 end
# manual_laser_frame_on = [788,780,778,789,786,788,789,775,788,776,787,786,787,786,778,781,772,789,786,776,775,780,787,787,785,787,786,784,781,787,786,790,788,780,784,788,785,780,789,790,785,785,788,788,789,775,788,782,789,783,779,770,785,774,777,772,787,786,786,788,789,782,789,782,777,780,777,780,787,787,779,786,784,781,787,785,789,787,785,787,786,784,788,773,786,782,787,781,782,784,787,788,788,779,774,786,789,782,787]  ## only good indication for legacy data
## maybe epsp here?
# experiments =           [365,371,373,378,380,382,386]
# manual_laser_frame_on = [780,775,776,778,772,786,787]

# experiments =           [372,375,376,378,380,383,384,387,388,390]       ## 462 end
# manual_laser_frame_on = [788,786,787,778,772,776,775,787,785,786]

# experiments =           [373,374,377,381,386,389,395,398,400]       ## 462 end
# manual_laser_frame_on = [776,787,786,789,787,787,790,784,785]

# experiments =           [366,367,368,371,373,374,377,379,381,382,385,386,392,393,394,395,396,397,398,399,400,401,402,405,410,416,426,427,429,430,433,434,436,437,439,440,442,444,445,447,452,457,458,459]
# manual_laser_frame_on = [778,789,786,775,776,787,786,781,789,786,780,787,781,787,786,790,788,780,784,788,785,780,789,785,788,785,789,782,780,777,787,779,784,781,785,789,785,786,784,773,782,779,774,786]


# path = '/media/jeremy/Data/local/Data_manip/2020_03_02/'
# experiments =           [131]
# manual_laser_frame_on = [100]




# path = args.main_folder_path
# experiments = range(args.experiments[0], args.experiments[1]+1)


## index for numpy array
dlp_on = 0
dlp_off = 1
laser_on = 2
laser_off = 3

rois_signal = []
for experiment in experiments:
    print(f'working on: {path}experiment_{experiment}')
    rois_signal.append(np.load(f'{path}experiment_{experiment}/dF_over_F0_backcorrect.npy'))

for exp_nb in range(len(rois_signal)):
    rois_signal[exp_nb][-1] = (0,0,manual_laser_frame_on[exp_nb],0)



signal_length = []
all_laser_on_value_on_x = []
takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))

for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
    for trace in range(len(rois_signal[0])-2):  ## the last one is the dlp and laser timings and before that is the x_axis data
        signal_length.append(len(rois_signal[exp_nb][trace]))
        all_laser_on_value_on_x.append(rois_signal[exp_nb][-1][laser_on]) ## not for legacy data


    min_signal_length = np.min(signal_length)
    max_laser_on_value_on_x = np.max(all_laser_on_value_on_x) ## not for legacy data
    x_axis = rois_signal[0][-2]
    value_to_center_on = takeClosest(max_laser_on_value_on_x, x_axis)
    x_axis_index_for_centering = x_axis.index(value_to_center_on)

shift = []
new_length = []
for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
    # pdb.set_trace()
    laser_on_value_on_x = rois_signal[exp_nb][-1][laser_on]
    shift.append(x_axis_index_for_centering - x_axis.index(takeClosest(laser_on_value_on_x, x_axis)))
    new_length.append(len(rois_signal[exp_nb][trace]) + shift[exp_nb])

for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
    for trace in range(len(rois_signal[0])-2):  ## the last one is the dlp and laser timings and before that is the x_axis data
        start_padding_array = np.zeros((shift[exp_nb]))
        rois_signal[exp_nb][trace] = np.insert(rois_signal[exp_nb][trace], 0, start_padding_array)
        # end_padding_array = np.zeros(([np.max(shift) - shift[i] for i in range(len(shift))][exp_nb]))
        end_padding_array = np.zeros((max(new_length) - len(rois_signal[exp_nb][trace])))
        rois_signal[exp_nb][trace] = np.insert(rois_signal[exp_nb][trace], -1, end_padding_array)
        # print([len(rois_signal[exp_nb][trace])for trace in range(len(rois_signal[0])-2)])
        # print(f'rois signal length: {len(rois_signal[exp_nb][trace])}')
    ## replace dlp times and laser times with new times taking padding into consideration
    laser_on_index = x_axis.index(takeClosest(rois_signal[exp_nb][-1][laser_on], x_axis)) + shift[exp_nb]

    _lst = list(rois_signal[exp_nb][-1])
    _lst[2] = x_axis[laser_on_index]

    rois_signal[exp_nb][-1] = tuple(_lst)

# normalize traces
if normalize:
    new_rois_signal = rois_signal
    for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
        for trace in range(len(rois_signal[0])-2):
            mu = np.mean(rois_signal[exp_nb][trace])
            sigma = np.std(rois_signal[exp_nb][trace])
            new_rois_signal[exp_nb][trace] = [((x - mu)/sigma) for x in rois_signal[exp_nb][trace]]
    rois_signal = new_rois_signal

averaged_rois_signal = np.zeros(((len(rois_signal[0])-2, len(rois_signal[0][0]))))
for trace in range(len(rois_signal[0])-2):
    _traces_for_average = np.zeros((len(rois_signal), len(rois_signal[0][0])))
    for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
        _traces_for_average[exp_nb] = rois_signal[exp_nb][trace]
    averaged_rois_signal[trace] = np.mean(_traces_for_average, axis=0)




def moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

rois_signal_moving_average = []
moving_average_points = 3
for trace in range(len(averaged_rois_signal)):
    moving_average_data = np.zeros((1, len(averaged_rois_signal[0])))
    moving_average_data = moving_average(averaged_rois_signal[trace],
                                n = moving_average_points) ## calculate moving average
    padding = np.zeros((len(rois_signal[exp_nb][trace]) - len(moving_average_data))) ##padding the start for plot
    moving_average_data = np.insert(moving_average_data, 0, padding)
    rois_signal_moving_average.append(moving_average_data)

rois_signal_per_experiment_moving_average = copy.deepcopy(rois_signal)
for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
    for trace in range(len(rois_signal[0])-2):
        moving_average_data = np.zeros((1, len(rois_signal[0][0])))
        moving_average_data = moving_average(rois_signal[exp_nb][trace],
                                    n = moving_average_points)
        padding = np.zeros((len(rois_signal[exp_nb][trace]) - len(moving_average_data)))
        moving_average_data = np.insert(moving_average_data, 0, padding)
        rois_signal_per_experiment_moving_average[exp_nb][trace] = moving_average_data



fig = go.Figure()
for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
    print(f'experiment_{experiment}')
    for trace in range(len(rois_signal[0])-2):  ## the last one is the dlp and laser timings and before that is the x_axis data
        print(f'trace_{trace}')
        fig.add_trace(go.Scatter(
                        x = rois_signal[0][-2],
                        y = rois_signal[exp_nb][trace],
                        name = f'experiment {experiment}-neuron {trace}',
                        #line_color = color[experiment],
                        opacity=0.5))
        fig.add_trace(go.Scatter(
                        x = rois_signal[0][-2],
                        y = rois_signal_per_experiment_moving_average[exp_nb][trace],
                        name = f'experiment {experiment}-neuron {trace} - moving average',
                        #line_color = color[experiment],
                        opacity=0.5))
    # fig.add_shape(type = "rect", xref = 'x', yref = 'paper',  ## dlp activation
    #                 x0 = rois_signal[exp_nb][-1][dlp_on], y0 = 0, x1 = rois_signal[exp_nb][-1][0][dlp_off], y1 = 1,  ## dlp
    #                 fillcolor="LightSkyBlue", opacity = 0.1, layer = 'below', line_width = 1)
    # fig.add_shape(type = "rect", xref = 'x', yref = 'y',  ## dlp activation
    #                 x0 = rois_signal[exp_nb][-1][laser_on], y0 = 30, x1 = rois_signal[exp_nb][-1][0][laser_off], y1 = 31,  ## laser
    #                 fillcolor="darksalmon", opacity = 0.05, layer = 'below', line_width = 1)
for trace in range(len(rois_signal[0])-2):  ## the last one is the dlp and laser timings and before that is the x_axis data
    fig.add_trace(go.Scatter(
                    x = rois_signal[0][-2],
                    y = averaged_rois_signal[trace],
                    name = f'trace {trace}- average',
                    #line_color = color[experiment],
                    opacity=1))
    fig.add_trace(go.Scatter(
                x = rois_signal[0][-2][0:len(rois_signal_moving_average[0])],
                y = rois_signal_moving_average[trace],
                name = f'trace {trace}- average using moving average',
                #line_color = color[experiment],
                opacity=0.5))
fig.show()
