import numpy as np
import plotly.graph_objects as go
import argparse
import pdb
import copy


def interactive_plotting(input_data_folder, experiment,
                        timing = True):

    #import pdb; pdb.set_trace()
    #classical_ephy = False
    #if classical_ephy:
    #    from classical_ephy import import_ephy_data

    #### ONLY ENTER EXPERIMENTS WITH NO TIMINGS PROBLEMS
    experiments = [experiment]
    # experiments = [[experiment[i]] for i in range(len(experiment))]
    # experiments = [89]
    # experiments = [131,132,133,141,142,148]

    #if multiple experiments
    if len(experiments) > 1:
        normalize = True
    else:
        normalize = False


    # input_data_folder = args.main_folder_path
    # experiments = range(args.experiments[0], args.experiments[1]+1)


    ## index for numpy array
    dlp_on = 0
    dlp_off = 1
    laser_on = 2
    laser_off = 3

    rois_signal = []
    #if classical_ephy:
    #    ephy_signal = []

    for experiment in experiments:
        # experiment=89
        # experiment = experiment[0]
        print(f'working on: {input_data_folder}/{experiment}')
        rois_signal.append(np.load(f'{input_data_folder}/{experiment}/dF_over_F0_backcorrect.npy', allow_pickle=True))
        #if classical_ephy:
        #    ephy_signal.append(import_ephy_data(input_data_folder, experiment))


    signal_length = []
    all_dlp_on_value_on_x = []
    takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))

                #####################################################
                ##### TO CHANGE IN UPDATED PIPELINE VERSION #########
    for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
        for trace in range(len(rois_signal[0])-2):  ## the last one is the dlp and laser timings and before that is the x_axis data
            signal_length.append(len(rois_signal[exp_nb][trace]))
            all_dlp_on_value_on_x.append(rois_signal[exp_nb][-1][0][dlp_on])

        min_signal_length = np.min(signal_length)
        max_dlp_on_value_on_x = np.max(all_dlp_on_value_on_x)
        x_axis = rois_signal[0][-2]
        value_to_center_on = takeClosest(max_dlp_on_value_on_x, x_axis)
        x_axis_index_for_centering = x_axis.index(value_to_center_on)

    shift = []
    new_length = []
    for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
        # pdb.set_trace()
        dlp_on_value_on_x = rois_signal[exp_nb][-1][0][dlp_on]
        shift.append(x_axis_index_for_centering - x_axis.index(takeClosest(dlp_on_value_on_x, x_axis)))
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
        dlp_on_index = x_axis.index(takeClosest(rois_signal[exp_nb][-1][0][dlp_on], x_axis)) + shift[exp_nb] ## dlp on index after padding
        dlp_off_index = x_axis.index(takeClosest(rois_signal[exp_nb][-1][0][dlp_off], x_axis)) + shift[exp_nb]
        laser_on_index = x_axis.index(takeClosest(rois_signal[exp_nb][-1][0][laser_on], x_axis)) + shift[exp_nb]
        laser_off_index = x_axis.index(takeClosest(rois_signal[exp_nb][-1][0][laser_off], x_axis)) + shift[exp_nb]

        #adjusting x_axis
        # padding_length = len(start_padding_array) + len(end_padding_array)
        # frame_time_difference = x_axis[-2] - x_axis[-3]
        #
        # len(x_axis)
        # [x_axis.append(x_axis[-2] + frame_time_difference) for _ in range(padding_length)]
        # len(x_axis)

        _lst = list(rois_signal[exp_nb][-1][0])
        _lst[0] = x_axis[dlp_on_index]
        _lst[1] = x_axis[dlp_off_index]
        _lst[2] = x_axis[laser_on_index]
        _lst[3] = x_axis[laser_off_index]

        rois_signal[exp_nb][-1][0] = tuple(_lst)
    # normalize traces
    if normalize:
        new_rois_signal = rois_signal
        for experiment, exp_nb in zip(experiments, range(len(rois_signal))):
            for trace in range(len(rois_signal[0])-2):
                mu = np.mean(rois_signal[exp_nb][trace])
                sigma = np.std(rois_signal[exp_nb][trace])
                new_rois_signal[exp_nb][trace] = [((x - mu)/sigma) for x in rois_signal[exp_nb][trace]]
        rois_signal = new_rois_signal
################################################################################################################################
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
    moving_average_points = 5
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
                                            #####################################################
                                            ##### TO CHANGE IN UPDATED PIPELINE VERSION #########
        fig.add_shape(type = "rect", xref = 'x', yref = 'paper',  ## dlp activation
                        x0 = rois_signal[exp_nb][-1][0][dlp_on], y0 = 0, x1 = rois_signal[exp_nb][-1][0][dlp_off], y1 = 1,  ## dlp
                        fillcolor="LightSkyBlue", opacity = 0.1, layer = 'below', line_width = 1)
        fig.add_shape(type = "rect", xref = 'x', yref = 'y',  ## dlp activation
                        x0 = rois_signal[exp_nb][-1][0][laser_on], y0 = 30, x1 = rois_signal[exp_nb][-1][0][laser_off], y1 = 31,  ## laser
                        fillcolor="darksalmon", opacity = 0.05, layer = 'below', line_width = 1)
        ####################################################################################################################

    # for trace in range(len(rois_signal[0])-2):  ## the last one is the dlp and laser timings and before that is the x_axis data
    #     fig.add_trace(go.Scatter(
    #                     x = rois_signal[0][-2],
    #                     y = averaged_rois_signal[trace],
    #                     name = f'trace {trace}- average',
    #                     #line_color = color[experiment],
    #                     opacity=1))
    #     fig.add_trace(go.Scatter(
    #                 x = rois_signal[0][-2][0:len(rois_signal_moving_average[0])],
    #                 y = rois_signal_moving_average[trace],
    #                 name = f'trace {trace}- average using moving average',
    #                 #line_color = color[experiment],
    #                 opacity=0.5))
    fig.write_html(f"{input_data_folder}/{experiment}/interactive_figure.html")
    # fig.show()




def interactive_plotting_no_timing(input_data_folder, experiment,
                                   restrict=False):
    # experiments = [experiment for experiment in experiment]
    experiments = [experiment]
    if len(experiments) > 1:
        normalize = True
    else:
        normalize = False

    ## index for numpy array
    dlp_on = 0
    dlp_off = 1
    laser_on = 2
    laser_off = 3

    rois_signal = []

    for experiment in experiments:
        print(f'working on: {input_data_folder}/{experiment}')
        rois_signal.append(np.load(f'{input_data_folder}/{experiment}/dF_over_F0_backcorrect.npy', allow_pickle=True))

    x_axis = rois_signal[0][-2]

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
    moving_average_points = 5
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
                            opacity=0.5))
            fig.add_trace(go.Scatter(
                            x = rois_signal[0][-2],
                            y = rois_signal_per_experiment_moving_average[exp_nb][trace],
                            name = f'experiment {experiment}-neuron {trace} - moving average',
                            opacity=0.5))
    fig.write_html(f"{input_data_folder}/{experiment}/interactive_figure.html")


if __name__ == "__main__":

    experiment = 'experiment_41'
    input_data_folder = f'/home/jeremy/Desktop/2020_11_23'

    interactive_plotting(input_data_folder, experiment)
    # interactive_plotting_no_timing(input_data_folder, experiment)
