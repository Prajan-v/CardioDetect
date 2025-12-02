import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('reports/images', exist_ok=True)

def test_plot():
    print("Generating test plot...")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title("Test Plot")
    plt.savefig('reports/images/test_plot.png')
    plt.close()
    print("✅ Generated reports/images/test_plot.png")

if __name__ == "__main__":
    test_plot()
