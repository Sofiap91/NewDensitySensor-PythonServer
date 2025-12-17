class s11_plot {
    constructor(div_id, frequencies, data) {
        var ctx = document.getElementById(div_id);
        var style = getComputedStyle(document.body);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: frequencies,
                datasets: [
                    {
                        label: 'S11 (real)',
                        data: chart_data_list_re_im(data, 'real'),
                        borderColor: style.getPropertyValue('--bs-primary'),
                        fill: false,
                        radius: 0,
                        order: 1
                    }, 
                    {
                        label: 'S11 (imag)',
                        data: chart_data_list_re_im(data, 'imag'),
                        borderColor: style.getPropertyValue('--bs-success'),
                        fill: false,
                        radius: 0,
                        order: 2
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'S11'
                    },
                    legend: {
                        display: true,
                        position: 'bottom'
                    },
                },
                tooltips: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(tooltipItem, data) {
                            var label = data.datasets[tooltipItem.datasetIndex].label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += tooltipItem.yLabel;
                            return label;
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Frequency (GHz)'
                        },
                        ticks: {
                            callback: function(value, index, values) {
                                return this.getLabelForValue(value) / 1e9;
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'S11'
                        },
                    }
                },
                elements: {
                    point:{
                        radius: 0
                    }
                }
            }
        });
    }

    updateData(data) {
        this.chart.data.datasets[0].data = chart_data_list_re_im(data, 'real');
        this.chart.data.datasets[1].data = chart_data_list_re_im(data, 'imag');

        this.chart.update();
    }
}

class s11_mag_phase_plot {
    constructor(div_id, frequencies) {
        var ctx = document.getElementById(div_id);
        var style = getComputedStyle(document.body);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: frequencies,
                datasets: [
                    {
                        label: 'abs(S11)',
                        data: [],
                        borderColor: style.getPropertyValue('--bs-primary'),
                        fill: false,
                        radius: 0,
                        order: 1,
                        yAxisId: 'y'
                    },
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'S11 - Return Loss'
                    },
                    legend: {
                        display: true,
                        position: 'bottom'
                    },
                },
                stacked: false,
                tooltips: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(tooltipItem, data) {
                            var label = data.datasets[tooltipItem.datasetIndex].label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += tooltipItem.yLabel;
                            return label;
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Frequency (GHz)'
                        },
                        ticks: {
                            callback: function(value, index, values) {
                                return this.getLabelForValue(value) / 1e9;
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'abs(S11) db',
                            position: 'left',
                            type: 'linear',
                        },
                        beginAtZero: true,
                        min: -30,
                    }
                },
                elements: {
                    point:{
                        radius: 0
                    }
                },
                grid: {
                    drawOnChartArea: false, // only want the grid lines for one axis to show up
                  },
            }
        });
    }

    updateData(data) {
        this.chart.data.datasets[0].data = chart_data_list_re_im(data, 'mag');

        this.chart.update();
    }
}

class stability_plot {
    constructor(div_id, frequencies, data) {
        var ctx = document.getElementById(div_id);
        var style = getComputedStyle(document.body);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: frequencies,
                datasets: [
                    {
                        label: 'abs(ΔS11) (db)',
                        data: [],
                        borderColor: style.getPropertyValue('--bs-primary'),
                        fill: false,
                        radius: 0,
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'ΔS11'
                    },
                    legend: {
                        display: true,
                        position: 'bottom'
                    },
                },
                tooltips: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(tooltipItem, data) {
                            var label = data.datasets[tooltipItem.datasetIndex].label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += tooltipItem.yLabel;
                            return label;
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Frequency (GHz)'
                        },
                        ticks: {
                            callback: function(value, index, values) {
                                return this.getLabelForValue(value) / 1e9;
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'abs(ΔS11) (db)'
                        },
                        beginAtZero: true,
                        min: -50,
                    }
                },
                elements: {
                    point:{
                        radius: 0
                    }
                }
            }
        });
    }

    updateData(data) {
        this.chart.data.datasets[0].data = chart_data_list(data);

        this.chart.update();
    }
}

function chart_data_list_re_im(inputData, selection) {
    dataList = []
    $.each(inputData, function (index, element) {
        dataList.push({ x: parseFloat(index), y: element[selection] });
    })
    return dataList;
}

function chart_data_list(inputData, selection) {
    dataList = []
    $.each(inputData, function (index, element) {
        dataList.push({ x: parseFloat(index), y: element });
    })
    return dataList;
}