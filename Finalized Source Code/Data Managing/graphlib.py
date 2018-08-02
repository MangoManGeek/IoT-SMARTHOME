import matplotlib.pyplot as plt
import seaborn as sns

#plots two 1D lists in the same figure using matplotlib
def plot(data_1, data_2):
    plt.figure(1)
    plt.plot(data_1)
    plt.plot(data_2)
    plt.show()


def comparisonPlot():
    plt.figure(1)
    real_data= extract_data_from_csv("")
    fake_data = generate_fake(real_data)

    i=1
    points = getDataAtIndex(real_data,i)
    plt.plot(points)
    points = getDataAtIndex(fake_data,i)
    plt.plot(points)
    plt.show()

def distPlot(costs_falsified,costs_unfalsified):
    sns.set(color_codes=True)
    sns.distplot(costs_falsified)
    sns.distplot(costs_unfalsified)
    plt.show()
