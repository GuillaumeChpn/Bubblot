(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('graphCtrl', graphCtrl);

    function graphCtrl($scope, $element) {
        var vm = this;
        var labels = [];
        for (var i = 0; i < 10; i++) {
            switch (i) {
                case 1:
                    labels.push(String("Cu"));
                    break;
                case 3:
                    labels.push(String("Fe"));
                    break;
                case 6:
                    labels.push(String("Sn"));
                    break;
                case 8:
                    labels.push(String("Pb"));
                    break;
                default:
                    labels.push(String(""));
            }
        } 
        var barData=[];
        for (var i = 0; i <= 10; i++) {
            //var obj = { x: 0.2 * i, y: 100 * Math.random() };
            if(i==1 || i==3 || i==6 || i==8) var obj = 100;
            else var obj = 0;
            barData.push(obj);
        }
        var initData = {
            labels: labels,
            datasets: [{
                backgroundColor: 'red',
                borderColor: 'red',
                data: barData,
            },{
                borderColor: 'black',
                pointBackgroundColor: 'black',
                borderWidth: 2,
                pointRadius: 1,
                data: [0,0,0,70,0,0,0,0,90,0],
                type:'line'
            }]
        };
        var context = $element.find("canvas")[0].getContext("2d");
        var graph = new Chart(context, {
            type: 'bar',
            data: initData,
            options: {
                maintainAspectRatio: false,
                legend: {
                    display: false,
                },
                tooltips: {
                    enabled: false,
                },
                scales: {
                    xAxes: [{
                        //type: 'linear',
                        position: 'bottom',
                        barPercentage : 0.15,
                        scaleLabel: {
                            display: false,
                            fontColor: "black",
                            fontSize: 10,
                            labelString: '[mV]',
                        },
                        ticks: {
                            //min: 0,
                            //max: 10,
                            //stepSize: 0.2,
                            fontColor: "black",
                            maxRotation: 0,

                        },
                        gridLines: {
                            display: true,
                            color: "black",
                        }
                    }],
                    yAxes: [{
                        type: 'linear',
                        position: 'left',
                        scaleLabel: {
                            display: false,
                            fontColor: "black",
                            fontSize: 10,
                            labelString: '[mV*2]',
                        },
                        ticks: {
                            min: 0,
                            max: 100,
                            fontColor: "black"
                        },
                        gridLines: {
                            display: true,
                            color: "black",
                        }
                    }]
                }
            }
        });
        //Update the information graph when clicking on map
        $scope.updateGraph = function () {
            /*
            $scope.data = [];
            for (var i = 0; i <= 10; i++) {
                //var obj = { x: 0.2 * i, y: 100 * Math.random() };
                var obj = 100 * Math.random();
                $scope.data.push(obj);
            }
            graph.data.datasets[1].data = $scope.data;
            graph.update();
            */
        };
    }
}());
