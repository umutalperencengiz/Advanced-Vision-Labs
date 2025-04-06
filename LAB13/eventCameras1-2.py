
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def load_process_events(path):
    events = []
    
    with open(path, 'r') as file:
        line = file.readline()
        while line:
            # Split the line into components
            timestamp, x, y, polarity = line.split()
            timestamp = float(timestamp)
            
            # Only save events where timestamp < 1
            if timestamp < 1:
                events.append([timestamp, int(x), int(y), int(polarity)])
            else:
                break
            
            # Read the next line
            line = file.readline()
    
    return events
def split_events(events):
    timestamps = []
    x_coords = []
    y_coords = []
    polarities = []
    
    for event in events:
        timestamps.append(event[0])
        x_coords.append(event[1])
        y_coords.append(event[2])
        polarities.append(event[3])
    
    return timestamps, x_coords, y_coords, polarities
def analyze_events(timestamps, x_coords, y_coords, polarities):
    # Number of events
    num_events = len(timestamps)
    print(f"Number of events: {num_events}")

    # First and last timestamp
    first_timestamp = timestamps[0]
    last_timestamp = timestamps[-1]
    print(f"First timestamp: {first_timestamp}")
    print(f"Last timestamp: {last_timestamp}")

    # Maximum and minimum values of pixel coordinates
    max_x = max(x_coords)
    min_x = min(x_coords)
    max_y = max(y_coords)
    min_y = min(y_coords)
    print(f"Max x coordinate: {max_x}, Min x coordinate: {min_x}")
    print(f"Max y coordinate: {max_y}, Min y coordinate: {min_y}")

    # Compare to image resolution 240x180
    print("Image resolution: 240x180")

    # Number of positive and negative polarity events
    positive_polarity = sum(1 for p in polarities if p > 0)
    negative_polarity = len(polarities) - positive_polarity
    print(f"Positive polarity events: {positive_polarity}")
    print(f"Negative polarity events: {negative_polarity}")

    # More events with positive or negative polarity
    if positive_polarity > negative_polarity:
        print("More positive polarity events.")
    else:
        print("More negative polarity events.")
def visualize_events_3Dchart(timestamps, x_coords, y_coords, polarities):
    # Split data into positive and negative polarity events
    positive_events = [(t, x, y) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities) if p > 0]
    negative_events = [(t, x, y) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities) if p <= 0]
    
    pos_t, pos_x, pos_y = zip(*positive_events) if positive_events else ([], [], [])
    neg_t, neg_x, neg_y = zip(*negative_events) if negative_events else ([], [], [])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot positive polarity events
    ax.scatter(pos_x, pos_y, pos_t, c='r', marker='o', label='Positive Polarity')
    
    # Plot negative polarity events
    ax.scatter(neg_x, neg_y, neg_t, c='b', marker='^', label='Negative Polarity')

    # Labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Timestamp')

    # Title and legend
    plt.title('3D Scatter Plot of Events')
    ax.legend()

    # Show plot
    plt.show()
def plot_3d_chart_subset(events, start_idx, end_idx, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the subset of events
    subset = events[start_idx:end_idx]

    # Split the data
    timestamps, x_coords, y_coords, polarities = zip(*subset)
    pos_events = [(t, x, y) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities) if p > 0]
    neg_events = [(t, x, y) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities) if p <= 0]

    pos_t, pos_x, pos_y = zip(*pos_events) if pos_events else ([], [], [])
    neg_t, neg_x, neg_y = zip(*neg_events) if neg_events else ([], [], [])

    # Plot positive polarity events
    ax.scatter(pos_x, pos_y, pos_t, c='r', marker='o', label='Positive Polarity')

    # Plot negative polarity events
    ax.scatter(neg_x, neg_y, neg_t, c='b', marker='^', label='Negative Polarity')

    # Labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Timestamp')
    plt.title(title)
    ax.legend()

    # Rotate the chart for better view
    ax.view_init(elev=30, azim=30)

    plt.show()
def plot_filtered_events(events, start_time, end_time, title):
    filtered_events = [event for event in events if start_time <= event[0] < end_time]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Split the data
    timestamps, x_coords, y_coords, polarities = zip(*filtered_events)
    pos_events = [(t, x, y) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities) if p > 0]
    neg_events = [(t, x, y) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities) if p <= 0]

    pos_t, pos_x, pos_y = zip(*pos_events) if pos_events else ([], [], [])
    neg_t, neg_x, neg_y = zip(*neg_events) if neg_events else ([], [], [])

    # Plot positive polarity events
    ax.scatter(pos_x, pos_y, pos_t, c='r', marker='o', label='Positive Polarity')

    # Plot negative polarity events
    ax.scatter(neg_x, neg_y, neg_t, c='b', marker='^', label='Negative Polarity')

    # Labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Timestamp')
    plt.title(title)
    ax.legend()

    # Rotate the chart for better view
    ax.view_init(elev=30, azim=30)

    plt.show()

if __name__ =="__main__":
    eventTextFile = "events.txt"
    events = load_process_events(eventTextFile)
    timestamps, x_coords, y_coords, polarities = split_events(events)
    visualize_events_3Dchart(timestamps, x_coords, y_coords, polarities)
    analyze_events(timestamps, x_coords, y_coords, polarities)
    plot_3d_chart_subset(events, 0, 8000, '3D Scatter Plot of First 8000 Events')
    plot_filtered_events(events, 0.5, 1, '3D Scatter Plot of Events (0.5 <= Timestamp < 1)')



""""
# Question: How long is the sequence used during exercise 1.1 (in seconds)?
# Answer: nearly 1 second(0.999996) as we are considering timestamps < 1.

# Question: What's the resolution of event timestamps?
# Answer: Number of events: 72045 Positive polarity events: 31036 Negative polarity events: 41009

# Question: What does the time difference between consecutive events depend on?
# Answer: The time difference between consecutive events depends on the sensor's capture rate and the speed of the changes in the scene being recorded.

# Question: What does positive/negative event polarity mean?
# Answer: Positive event polarity indicates a positive change in the brightness of a pixel, while negative polarity indicates a negative change in the brightness.

# Question: What is the direction of movement of objects in exercise 1.2?
# Answer: More negative polarity events. Positive polarity events may indicate the direction towards brighter areas,
while negative polarity events indicate movement towards darker areas.

"""