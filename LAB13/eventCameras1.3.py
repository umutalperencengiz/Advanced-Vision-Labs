import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
def load_process_events(path):
    events = []
    
    with open(path, 'r') as file:
        line = file.readline()
        while line:
            # Split the line into components
            timestamp, x, y, polarity = line.split()
            timestamp = float(timestamp)
            
            # Only save events where timestamp < 1
            if  2 > timestamp > 1:
                events.append([timestamp, int(x), int(y), int(polarity)])
            elif timestamp>=2:
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
        if event[3] == 0:
            polarities.append(-1)
        else:
            polarities.append(events[3])
    
    return timestamps, x_coords, y_coords, polarities
def event_frame(event_coords, polarities, image_shape):
    image = (np.ones(image_shape) * 127).astype("uint8")
    for (x, y), polarity in zip(event_coords, polarities):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:  
            if polarity == 1:
                image[y, x] = 255  # Positive event
            elif polarity == -1:
                image[y, x] = 0  # Negative event

    return image
if __name__ =="__main__":
    eventTextFile = "events.txt"
    imageShape = cv2.imread("images\\frame_00000000.png").shape[:2]
    events = load_process_events(eventTextFile)
    timestamps, x_coords, y_coords, polarities = split_events(events)
    event_coords = list(zip(x_coords, y_coords))
    event_image = event_frame(event_coords, polarities,imageShape)
    cv2.imshow("Event Image For 1-2",event_image)
    cv2.waitKey(0)
    tau = 0.1  # 10 ms expressed in seconds
    
    start_index = 0
    num_events = len(timestamps)
    
    while start_index < num_events:
        end_index = start_index
        
        while end_index < num_events and (timestamps[end_index] - timestamps[start_index]) <= tau:
            end_index += 1
        
        aggregated_event_coords = list(zip(x_coords[start_index:end_index], y_coords[start_index:end_index]))
        aggregated_polarities = polarities[start_index:end_index]
        
        event_image = event_frame(aggregated_event_coords, aggregated_polarities, imageShape)
        
        cv2.imshow(f"Event Image within {tau}", event_image)
        cv2.waitKey(0)  # Wait for key press to show next frame
        
        start_index = end_index

    cv2.destroyAllWindows()
    # Comment on tau values:
# tau = 1ms results are much more noisy.
# tau = 10ms is a balanced choice, providing a good number of events per frame and a smooth visualization.
# tau = 100ms results in fewer frames with many events in each and strong images.