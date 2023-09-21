(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('nozeCtrl', nozeCtrl);

    function nozeCtrl($scope, $element) {
        var vm = this;
        var detectionElement = $element.find(".detection")[0];
        detectionElement.style.width = "15%";
        detectionElement.style.height = "15%";
        detectionElement.style.top = "15%";
        detectionElement.style.left = "30%";
        $scope.updateMagnetism = function (element) {
            detectionElement.style.left = $scope.magnetismXaxis + "%";
            detectionElement.style.top = $scope.magnetismYaxis + "%";
        }
        //var context = $element.find("canvas")[0].getContext("2d");
        /*
        var graph = new Chart(context, {
            type: 'scatter',
            data: [{
                x: 0.5,
                y: 0.5
            }],
            options: {
                maintainAspectRatio: false,
                legend: {
                    display: false,
                },
                tooltips: {
                    enabled: false,
                },
                responsive: true,
                scales: {
                    xAxes: [{
                        type: 'linear',
                        position: 'bottom',
                        scaleLabel: {
                            display: false,
                            fontColor: "black",
                            fontSize: 10,
                            labelString: '[m]',
                        },
                        ticks: {
                            min: -1,
                            max: 1,
                            stepSize: 0.5,
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
                            labelString: '[m]',
                        },
                        ticks: {
                            min: -1,
                            max: 1,
                            stepSize: 0.5,
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
        */
    }

}());
