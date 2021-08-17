import plotly.express as px
import plotly.offline as opy


def pie_plot_from_lists(labels=None, values=None, title=None):
    """
        Update the plot data in plotly format.

        :param labels: The labels of the pie chart.
        :param values: The values of the pie chart.
        :param plot_title: The title of the plotly plot.
        :return: A string containing the graph in html form.
        """

    fig = px.pie(values=values, names=labels,
                 title=title)
    div = opy.plot(fig, auto_open=False, output_type='div', include_plotlyjs=False)
    return div

def plotlify_pie_js(xy=None, description=""):
    """
    Update the plot data in plotly format.

    :param xy: x and y in a single structure.
    :param description: The description of the plotly plot.
    :param plot_type: The type of the plotly plot.
    :return: A dictionary with the data in plotly format.
    """

    ret = {
        'data': [],
        'layout': {
            'paper_bgcolor': 'rgba(243, 243, 243, 1)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'title': {
                'text': description,
            }
        }
    }

    ret['data'].append(
        {
            'values': [v for k, v in xy.items()],
            'labels': [str(k) for k, v in xy.items()],
            'type': 'pie',
        }
    )
    return ret


def plotlify_scatter_js(xy=None, x=None, y=None, xtag=None, ytag=None, description=""):
    """
    Update the plot data in plotly format.

    :param xy: x and y in a single structure.
    :param description: The description of the plotly plot.
    :param plot_type: The type of the plotly plot.
    :return: A dictionary with the data in plotly format.
    """

    ret = {
        'data': [],
        'layout': {
            'paper_bgcolor': 'rgba(243, 243, 243, 1)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'title': {
                'text': description,
            }
        }
    }
    if xy:
        ret['data'].append(
            {
                'x': [v for k, v in xy.items()],
                'y': [k for k, v in xy.items()],
                'type': 'scatter',
            }
        )
    elif x and y:
        ret['data'].append(
            {
                'x': x,
                'y': y,
                'type': 'scatter',
            }
        )
    return ret