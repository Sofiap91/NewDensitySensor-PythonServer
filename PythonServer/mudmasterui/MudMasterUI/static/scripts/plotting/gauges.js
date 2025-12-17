const gaugePlugin = {
    beforeDraw: function(chart, args, options) {
        var width = chart.width,
            height = chart.height,
            ctx = chart.ctx;

        ctx.restore();
        var fontSize = (height / 114).toFixed(2);
        ctx.font = fontSize + "em sans-serif";
        ctx.textBaseline = "bottom";

        var text = chart.config._config.data.datasets[0].data[0] + chart.config._config.options.elements.units,
            textX = Math.round((width - ctx.measureText(text).width) / 2),
            textY = (height * 3 / 4).toFixed(2);

        ctx.fillText(text, textX, textY);
        ctx.save();
    }
};

class Gauge {
    constructor(id, label, min, max, value, units) {
        var ctx = document.getElementById(id);
        var style = getComputedStyle(document.body);

        this.chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [value, (max - value)],
                    backgroundColor: [
                        style.getPropertyValue('--bs-primary'),
                        style.getPropertyValue('--bs-light')
                    ],
                    borderWidth: 0
                }],
                labels: [label, units]
            },
            options: {
                cutoutPercentage: 60,
                plugins: {
                    title: {
                        display: true,
                        text: label,
                        position: 'bottom',
                    },
                    legend: {
                        display: false,
                    },
                },
                circumference: 180,
                rotation: -90,
                animation: {
                    animateScale: true,
                    animateRotate: true
                },
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    units: units,
                    maxValue: max,
                    minValue: min
                }
            },
            plugins: [{
              id: 'customPlugin',
              beforeDraw: gaugePlugin.beforeDraw,
            }]
        });
    }

    setValue(value) {
        this.chart.data.datasets[0].data[0] = value;
        this.chart.data.datasets[0].data[1] = this.chart.options.elements.maxValue - this.chart.data.datasets[0].data[0];
        this.chart.update();
    }
}