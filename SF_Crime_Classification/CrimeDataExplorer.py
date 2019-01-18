import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import sys
import os
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler


class CrimeDataExplorer(object):
    def __init__(self, train_raw_data, project_dir):
        self.train_data = pd.read_csv(train_raw_data, index_col=['Id'], parse_dates=['Dates'])
        self.images_path = project_dir

    def explore_data(self):
        self.train_data['Dates'] = pd.to_datetime(self.train_data['Dates'])
        self.train_data['Hour'] = self.train_data['Dates'].map(lambda x: x.hour)
        self.train_data['Month'] = self.train_data['Dates'].map(lambda x: x.month)
        self.train_data['Year'] = self.train_data['Dates'].map(lambda x: x.year)
        self.train_data['DayOfMonth'] = self.train_data['Dates'].map(lambda x: x.day)
        self.crime_histogram()
        self.show_crimes_by_group('Hour')
        self.show_crimes_by_group('Month')
        self.show_crimes_by_group('Year')
        self.show_crimes_by_group('DayOfWeek')
        self.show_crimes_by_group('PdDistrict')
        self.show_crimes_by_location()

    def crime_histogram(self):
        crimes_rating = self.train_data['Category'].value_counts()
        y_pos = np.arange(len(crimes_rating[0:18].keys()))
        plt.rcParams['figure.figsize'] = (20.0, 10.0)
        plt.barh(y_pos, crimes_rating[0:18].get_values(), align='center', alpha=0.4, color='black')
        plt.yticks(y_pos, map(lambda x: x.title(), crimes_rating[0:18].keys()), fontsize=12)
        plt.xlabel('Number of occurences', fontsize=14)
        plt.title('San Franciso Crimes', fontsize=28)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(self.images_path+"/crime_histogram.png")
        plt.close()

    def show_crimes_by_group(self, group_name):
        crimes_rating = self.train_data['Category'].value_counts()
        print("Running for Group",group_name)
        pylab.rcParams['figure.figsize'] = (24.0, 50.0)
        with plt.style.context('fivethirtyeight'):
            ax1 = plt.subplot2grid((13, 3), (0, 0), colspan=3)
            ax1.plot(self.train_data.groupby(group_name).size(), 'ro-')
            ax1.set_title('All crimes')
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks(np.arange(start, end, 1))
            x_grid = 1
            y_grid = 0
            for crime in crimes_rating.index.tolist():
                if y_grid == 3:
                    x_grid = x_grid + 1
                    y_grid = 0
                filtered_data = self.train_data[self.train_data['Category'] == crime]
                ax2 = plt.subplot2grid((13, 3), (x_grid, y_grid))
                ax2.plot(filtered_data.groupby(group_name).size(), 'o-')
                ax2.set_title(crime)
                y_grid += 1
            pylab.gcf().text(0.5, 1.03,
                             'San Franciso Crime Occurence by '+group_name,
                             horizontalalignment='center',
                             verticalalignment='top',
                             fontsize=28)
        plt.tight_layout(1)
        plt.savefig(self.images_path+"/crimes_by_"+group_name+".png")
        plt.close()

    def get_correlation_plot(self, processed_train_data):
        sns.set(style="white")
        plt.figure(figsize=(36, 20))
        sns.heatmap(processed_train_data.corr(), square=True, linewidths=.3)
        plt.savefig(self.images_path + "/features_correlaton.png")
        plt.close()

    def show_crimes_by_location(self):
        xy_scaler = StandardScaler()
        xy_scaler.fit(self.train_data[["X", "Y"]])
        self.train_data[["X", "Y"]] = xy_scaler.transform(self.train_data[["X", "Y"]])
        self.train_data = self.train_data[abs(self.train_data["Y"]) < 100]
        self.train_data.index = range(len(self.train_data))
        plt.plot(self.train_data["X"], self.train_data["Y"], '.')
        plt.savefig(self.images_path+"/location_map.png")
        plt.close()
        NX = 100
        NY = 100
        groups = self.train_data.groupby('Category')
        ii = 1
        plt.figure(figsize=(20, 20))
        for name, group in groups:
            plt.subplot(8, 5, ii)
            histo, xedges, yedges = np.histogram2d(np.array(group.X), np.array(group.Y), bins=(NX, NY))
            myextent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(histo.T, origin='low', extent=myextent, interpolation='nearest', aspect='auto', norm=LogNorm())
            plt.title(name)
            ii += 1
        plt.savefig(self.images_path+"/crimes_by_location.png")
        plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", help="Path to TRAIN DATA FILE", required=True)
    parser.add_argument("-o", "--output", help="Output Directory For Plots and Saved Model", required=True)

    args = parser.parse_args()
    train_data_name = args.train
    plot_folder = args.output

    if not os.path.exists(train_data_name):
        print("Specified Data File doesnt exist")
        exit(1)
    if not os.path.exists(plot_folder):
        print("Output Directory doesnt exist. Creating it now")
        os.mkdir(plot_folder)
    explorer = CrimeDataExplorer(train_data_name, plot_folder)
    explorer.explore_data()
