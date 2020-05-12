import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
how to use:
python plotting_graphs.py \
--main_folder_path /media/jeremy/Data/local/Data_manip/2020_02_05/ \
--experiments 1 \
--experiments 10 \
--columns 10 \
--rows 10
"""

parser = argparse.ArgumentParser(description="options")
parser.add_argument("--main_folder_path", required=True, action="store",
                        help="main folder absolute path, somehting like /media/jeremy/Data/local/Data_manip/2020_02_05")
parser.add_argument("--experiments", required=True, action="append", type=int,
                        help="values of experiments to go over")
parser.add_argument("--columns", required=True, action="store", type=int,
                        help="number of columns of the resulting graph")
parser.add_argument("--rows", required=True, action="store", type=int,
                        help="number of rows of the resulting graph")
args = parser.parse_args()



path = args.main_folder_path
experiments = range(args.experiments[0], args.experiments[1]+1)
# experiments = range(250,281)
# experiments = [10,11,12,13,14,15,16,18,19,20,21]

graphs = 4
graph0 = 'pixel_time_serie_whole_data.png'
graph1 = 'pixel_time_serie_with_backg_substraction_whole_data.png'
graph2 = 'delta_F_over_F0__whole_data.png'
graph3 = 'delta_F_over_F0_ephys_style_whole_data.png'

# fig, ax = plt.subplots(len(experiments),4, figsize=(100,100))
# for experiment in range(len(experiments)):
#     images_path = f"{path}experiment_{experiments[experiment]}/"
#     graph = range(graphs)
#
#     ax[experiment,graph[0]].imshow(plt.imread(images_path+graph0))
#     ax[experiment,graph[0]].axis('off')
#
#     ax[experiment,graph[1]].imshow(plt.imread(images_path+graph1))
#     ax[experiment,graph[1]].axis('off')
#
#     ax[experiment,graph[2]].imshow(plt.imread(images_path+graph2))
#     ax[experiment,graph[2]].axis('off')
#
#     ax[experiment,graph[3]].imshow(plt.imread(images_path+graph3))
#     ax[experiment,graph[3]].axis('off')
#
# plt.savefig(f"{path}summary_figure_experiment_{experiments[0]}-{experiments[-1]}.png")
# plt.savefig(f"{path}summary_figure_experiment_{experiments[0]}-{experiments[-1]}.svg")

columns = args.columns #5
rows = args.rows #10
fig, ax = plt.subplots(rows, columns, figsize=(50,50), sharex='col', sharey='row')
experiment = 0
for row in range(rows):
    for column in range(columns):
        try:
            print(f'working on experiment - {experiments[experiment]}')
            images_path = f"{path}experiment_{experiments[experiment]}/"
            ax[row, column].imshow(plt.imread(images_path+graph2))
            ax[row, column].axis('off')
            ax[row, column].title.set_text(f'{experiments[experiment]}')
            fig.tight_layout()
            experiment += 1
            plt.savefig(f"{path}summary_figure_experiment_{experiments[0]}-{experiments[-1]}.png")
        except:
            print('no other plot')
