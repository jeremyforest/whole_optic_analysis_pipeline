import numpy as np
import matplotlib.pyplot as plt


path = "/media/jeremy/Data/local/Data_manip/2020_02_18/"
experiments = range(1,31)
# experiments = [10,11,12,13,14,15,16,18,19,20,21]

graphs = 4
graph0 = 'pixel_time_serie.png'
graph1 = 'pixel_time_serie_with_backg_substraction.png'
graph2 = 'delta F over F0.png'
graph3 = 'delta F over F0 ephys style.png'

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

columns = 5 #5
rows = 6 #10
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
