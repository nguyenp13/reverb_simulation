# Caribou Location Tracking Visualization

This is a visualization tracking the location of 260 woodland caribou in northern British Columbia from 1988 to 2016.

More information about the raw data can be found  [here](https://www.kaggle.com/jessemostipak/caribou-location-tracking).

A live demo can be found [here](http://paul-tqh-nguyen.github.io/caribou_location_tracking/).

Feel free to  [reach out](https://paul-tqh-nguyen.github.io/about/#contact)  for help or report problems or make suggestions for improvement!

### Tools Used

The following tools were heavily utilized to create this visualization:
* [Bokeh](https://bokeh.org/)
* [Pandas](https://pandas.pydata.org/)
* [Pandarallel](https://github.com/nalepae/pandarallel)

We also used [matplotlib]([https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xlim.html](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.xlim.html)), [datetime](https://docs.python.org/3/library/datetime.html), [numpy](https://numpy.org/), and [multiprocessing](https://docs.python.org/3/library/multiprocessing.html).

### Usage

Use the slider to select a date. The locations of the caribou on this date will be shown on the map.

The caribou will be shown as colored dots on the map.

Hover over the dots to get information about the caribou.

Dragging the slider may change dates in larger increments than desired, so using the "Previous Date" and "Next Date" buttons may be useful here.

The colored lines correspond to the paths of the caribou (though the caribou may not necessarily be present on the selected date).

Use the "Play" button to have the date slider automatically progress the date forward in time.

There are utilities near the upper right corner of the map that to pan, box zoom, scroll zoom, and reset the zoom level.
